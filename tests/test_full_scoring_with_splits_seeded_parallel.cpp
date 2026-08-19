// test_full_scoring_with_splits_seeded_parallel.cpp
//
// Combines test_full_scoring_with_splits_seeded.cpp (seeded eval keys AND
// seeded query, both reconstructed with NO secret key involved) with
// test_full_scoring_with_splits_parallel.cpp's memory-bounded threading
// pattern -- see that file's header for the full reasoning on why database
// construction happens per-(cluster,split) iteration rather than as one
// precomputed whole-database structure (peak memory bounded by thread
// count, not total pair count; ~4 GB instead of ~80 GB for a realistic
// parameter file).
//
// Notably simpler than the non-seeded version for the query-switching step:
// build_seeded_query/reconstruct_query are already CRT-aware (see
// seeded_query.hpp), so query.embedding_cts comes back as [ring][j]
// directly -- no manual per-ring encryption loop needed here at all, only
// the parallel switch+eval-form-conversion step, flattened over (ring, j)
// exactly as in the non-seeded version.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   OMP_NUM_THREADS=16 ./test_full_scoring_with_splits_seeded_parallel

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
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class FullScoringWithSplitsSeededParallel : public ::testing::TestWithParam<Params (*)()> {};

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

TEST_P(FullScoringWithSplitsSeededParallel, EachSplitProducesCorrectResultUsingSeededEvalKeysAndQuery) {
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

        // --- Eval keys: seeded, wire-compressed, reconstructed. UNCHANGED
        // by CRT -- no dependency on plaintext modulus at all. -----------------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- Query: sample embedding values, seed-compress (build_seeded_query
        // handles CRT-splitting internally), reconstruct. -------------------------------
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<int64_t> embedding_values(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            messages[static_cast<size_t>(j)] = sample_signed_value(params, rng);
            embedding_values[static_cast<size_t>(j)] = reduce_mod(messages[j].raw, combined_modulus);  // messages[static_cast<size_t>(j)].reduced;
        }

        SeededQuery query_wire =
            build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);
        ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);
        ASSERT_EQ(static_cast<int64_t>(query.embedding_cts.size()), r);

        // --- Switch reconstructed embedding ciphertexts to RLWE/eval form,
        // PER COMPONENT RING, using the RECONSTRUCTED pub, IN PARALLEL,
        // flattened over (ring, j). query.embedding_cts is ALREADY [ring][j]. ------
        std::vector<std::vector<std::unique_ptr<RLWECTEvalForm>>> query_eval(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            query_eval[static_cast<size_t>(ring)].resize(query.embedding_cts[static_cast<size_t>(ring)].size());
        }

        int64_t total_query_terms = r * params.embedding_length;
        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < total_query_terms; ++idx) {
            int64_t ring = idx / params.embedding_length;
            int64_t j = idx % params.embedding_length;
            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(
                rlwe_ct, query.embedding_cts[static_cast<size_t>(ring)][static_cast<size_t>(j)]);
            query_eval[static_cast<size_t>(ring)][static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
        }

        // --- Switch reconstructed selector ciphertexts to RGSW, using the
        // RECONSTRUCTED pub. UNCHANGED by CRT. -----------------------------------------
        std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(query.selector_cts.size());
        #pragma omp parallel for schedule(dynamic)
        for (int64_t c = 0; c < static_cast<int64_t>(query.selector_cts.size()); ++c) {
            RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]);
            rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
        }

        // --- Scoring: parallel over (cluster, split) ONLY -- see
        // test_full_scoring_with_splits_parallel.cpp's header for why this,
        // not a precomputed whole database, is what keeps memory bounded. --------------
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
        }

        // --- Sequential reduction: sum over clusters, per split, per ring. ------------
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
        // both rings, then recompose (a no-op when r==1) -- checked directly
        // against the desired cluster's raw values. -----------------------------------
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

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, FullScoringWithSplitsSeededParallel,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });