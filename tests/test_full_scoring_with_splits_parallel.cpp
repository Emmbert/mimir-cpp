// test_full_scoring_with_splits_parallel.cpp
//
// Same protocol and same correctness checks as
// test_full_scoring_with_splits.cpp, but RLWE switching, RGSW switching,
// database construction, and per-(cluster,split) scoring are parallelized
// via OpenMP -- the same threading pattern used in
// benchmarks/benchmark_latency_parallel.cpp.
//
// This is a correctness sanity check for that parallelization, not a new
// protocol test: the expected results are identical to the single-threaded
// version, computed with multiple threads.
//
// MEMORY, not just correctness, drives this file's structure. An earlier
// version of this test built the ENTIRE database up front, as one complete
// structure shared by an outer loop over component rings (each ring's
// parallel (cluster,split) loop simply reading its slice of that
// structure). That's fine at small test-parameter scale, but for a
// realistic parameter file (large embedding_length, hundreds of
// (cluster,split) pairs), holding every (cluster,split,ring) database
// polynomial's eval-form simultaneously runs to tens of gigabytes and gets
// the process OOM-killed. So this file does NOT precompute the whole
// database: parallelization happens over (cluster,split) pairs directly
// (not further split by ring), and EACH iteration samples its own data,
// derives BOTH rings' eval-form polynomials from that SAME sample (via
// crt_split_database_polynomial_eval_form -- preserving the CRT invariant
// that both rings represent the same underlying data), scores and masks
// both rings, and lets all of it go out of scope once that iteration is
// done. Peak memory is bounded by (thread count) x (one pair's data), not
// by the size of the whole database -- see the chat message this version
// was written in for the concrete estimate (roughly 4 GB instead of ~80 GB
// for a realistic parameter file).
//
// This also means verification only ever needs the DESIRED cluster's raw
// values -- every other cluster gets masked to zero regardless of its
// content, so its exact values never mattered for correctness checking.
// That's stashed (small: splits_per_cluster worth, not the whole
// database) the moment a (cluster,split) iteration happens to be for the
// desired cluster -- safe without synchronization, since desired_cluster_index
// is fixed, so exactly one c value across the whole run ever satisfies
// c == desired_cluster_index for a given s.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   OMP_NUM_THREADS=16 ./test_full_scoring_with_splits_parallel

#include <gtest/gtest.h>

#include <omp.h>

#include <memory>
#include <random>
#include <vector>

#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class FullScoringWithSplitsParallel : public ::testing::TestWithParam<Params (*)()> {};

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

/// score = sum_j query[j] * db[j], for ONE component ring, ONE
/// (cluster, split) pair. UNCHANGED from the non-CRT version.
RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<std::unique_ptr<RLWECTEvalForm>>& query_eval,
                            const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j]->mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

} // namespace

TEST_P(FullScoringWithSplitsParallel, MatchesSingleThreadedResultUnderMultithreading) {
    Params params = GetParam()();

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully.";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);
    /*if (params.splits_per_cluster == 1) {
        GTEST_SKIP() << "This test needs splits_per_cluster > 1 to exercise multiple splits; got "
                     << "cluster_size=" << params.cluster_size << ", n=" << params.n
                     << ", splits_per_cluster=" << params.splits_per_cluster
                     << ". Increase database_size or decrease num_clusters/n.";
    }*/

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single split's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;
    int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

    constexpr int kNumIterations = 3;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Query: sample + CRT-split + encrypt SEQUENTIALLY (touches
        // secret.lwe_sk's internal state), then key-switch to RLWE and
        // convert to eval form IN PARALLEL, flattened over (ring, j). --------------
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<std::vector<LWECT>> embedding_lwe(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            embedding_lwe[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.embedding_length));
        }
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages[static_cast<size_t>(j)] = m;

            std::vector<int64_t> components;
            if (r == 1) {
                components = {m.reduced};
            } else {
                int64_t canonical = reduce_mod(m.raw, combined_modulus);
                auto [c1, c2] = crt_split(canonical, params.comp_ring_modulus);
                components = {c1, c2};
            }
            for (int64_t ring = 0; ring < r; ++ring) {
                embedding_lwe[static_cast<size_t>(ring)].push_back(secret.lwe_sk->encode_and_encrypt(
                    components[static_cast<size_t>(ring)], ctx.component_encodings[static_cast<size_t>(ring)]));
            }
        }

        std::vector<std::vector<std::unique_ptr<RLWECTEvalForm>>> query_eval(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            query_eval[static_cast<size_t>(ring)].resize(static_cast<size_t>(params.embedding_length));
        }

        int64_t total_query_terms = r * params.embedding_length;
        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < total_query_terms; ++idx) {
            int64_t ring = idx / params.embedding_length;
            int64_t j = idx % params.embedding_length;
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, embedding_lwe[static_cast<size_t>(ring)][static_cast<size_t>(j)]);
            query_eval[static_cast<size_t>(ring)][static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
        }

        // --- Cluster-selection: UNCHANGED by CRT -- gadget-encrypt
        // sequentially, key-switch to RGSW in parallel, over clusters only. ------------
        std::vector<LWEGadgetCT> selector_gadget;
        selector_gadget.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            selector_gadget.push_back(secret.lwe_gadget_sk->gadget_encrypt(bit));
        }

        std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(static_cast<size_t>(params.num_clusters));
        #pragma omp parallel for schedule(dynamic)
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(selector_gadget[static_cast<size_t>(c)]);
            rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
        }

        // --- Scoring: parallel over (cluster, split) ONLY -- NOT flattened
        // or split by ring. Each iteration samples its own data ONCE and
        // uses it for BOTH rings, then discards it -- see file header for
        // why this, not a precomputed whole database, is what keeps memory
        // bounded. Only the desired cluster's raw values are kept, and only
        // for as long as this test needs them. -----------------------------------------
        std::vector<std::vector<std::vector<std::unique_ptr<RLWECT>>>> masked( // [s][ring][c]
            static_cast<size_t>(params.splits_per_cluster));
        for (auto& per_split : masked) {
            per_split.resize(static_cast<size_t>(r));
            for (auto& per_ring : per_split) {
                per_ring.resize(static_cast<size_t>(params.num_clusters));
            }
        }

        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values( // [s][j]
            static_cast<size_t>(params.splits_per_cluster));

        std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
        for (auto& tr : thread_rngs) tr.seed(rng());

        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < total_pairs; ++idx) {
            int64_t c = idx / params.splits_per_cluster;
            int64_t s = idx % params.splits_per_cluster;
            std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

            std::vector<std::vector<int64_t>> split_raw_values; // [j], only populated if c == desired
            if (c == params.desired_cluster_index) {
                split_raw_values.resize(static_cast<size_t>(params.embedding_length));
            }

            std::vector<std::vector<DatabasePolynomialEvalForm>> db_eval_per_j( // [j][ring]
                static_cast<size_t>(params.embedding_length));

            for (int64_t j = 0; j < params.embedding_length; ++j) {
                std::vector<int64_t> raw(static_cast<size_t>(params.n));
                for (int64_t i = 0; i < params.n; ++i) {
                    raw[static_cast<size_t>(i)] = sample_signed_value(params, local_rng).raw;
                }
                db_eval_per_j[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
                if (c == params.desired_cluster_index) {
                    split_raw_values[static_cast<size_t>(j)] = std::move(raw);
                }
            }

            if (c == params.desired_cluster_index) {
                // Safe without synchronization: desired_cluster_index is
                // fixed, so exactly one c value across the whole run ever
                // reaches this branch for a given s.
                desired_cluster_raw_values[static_cast<size_t>(s)] = std::move(split_raw_values);
            }

            for (int64_t ring = 0; ring < r; ++ring) {
                std::vector<DatabasePolynomialEvalForm> db_split_ring;
                db_split_ring.reserve(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    db_split_ring.push_back(db_eval_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)]);
                }

                RLWECT score = compute_split_score(ctx, query_eval[static_cast<size_t>(ring)], db_split_ring);

                RLWECT masked_val(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score); // SAME rgsw_ct for every ring

                masked[static_cast<size_t>(s)][static_cast<size_t>(ring)][static_cast<size_t>(c)] =
                    std::make_unique<RLWECT>(masked_val);
            }
            // db_eval_per_j (and split_raw_values, once moved from) go out
            // of scope here -- nothing from this iteration persists except
            // masked[s][ring][c] and, for exactly one c per s, the stashed
            // desired-cluster raw values.
        }

        // --- Sequential reduction: sum over clusters, per split, per ring.
        // Cheap relative to the parallel work above. ----------------------------------
        std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r)); // [ring][s]
        for (int64_t ring = 0; ring < r; ++ring) {
            final_result[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result[static_cast<size_t>(ring)].emplace_back(ctx.rlwe_param);
            }
        }
        for (int64_t ring = 0; ring < r; ++ring) {
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t c = 0; c < params.num_clusters; ++c) {
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)].add(
                        final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                        *masked[static_cast<size_t>(s)][static_cast<size_t>(ring)][static_cast<size_t>(c)]);
                }
            }
        }

        // --- Decrypt and verify EACH split's result independently, across
        // both rings, then recompose (a no-op when r==1). --------------------------
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            std::vector<Vector> decrypted_per_ring;
            decrypted_per_ring.reserve(static_cast<size_t>(r));
            for (int64_t ring = 0; ring < r; ++ring) {
                decrypted_per_ring.push_back(secret.rlwe_sk->decrypt_vector(
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                    ctx.component_encodings[static_cast<size_t>(ring)]));
            }

            const auto& raw_values_for_split = desired_cluster_raw_values[static_cast<size_t>(s)];

            for (int64_t i = 0; i < params.n; ++i) {
                int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                               : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                                params.comp_ring_modulus);

                int64_t true_sum = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    true_sum += messages[static_cast<size_t>(j)].raw *
                                raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)];
                }

                int64_t expected = reduce_mod(true_sum, combined_modulus);
                EXPECT_EQ(recomposed, expected)
                    << "Mod-p mismatch on iteration " << iter << ", split " << s << ", coefficient " << i
                    << " (r=" << r << "): expected=" << expected << " got=" << recomposed;

                int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
                EXPECT_EQ(decoded_signed, true_sum)
                    << "Overflow/corruption on iteration " << iter << ", split " << s << ", coefficient " << i
                    << " (r=" << r << "): true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                    << " (recomposed residue was " << recomposed << ").";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, FullScoringWithSplitsParallel,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });