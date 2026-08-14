// test_full_scoring_with_splits_seeded_parallel.cpp
//
// Combines test_full_scoring_with_splits_seeded.cpp (seeded eval keys AND
// seeded query, both reconstructed with NO secret key involved in
// reconstruction) with test_full_scoring_with_splits_parallel.cpp's
// threading pattern (RLWE switching, RGSW switching, and per-(cluster,split)
// scoring all parallelized via OpenMP).
//
// This is the closest thing so far to what a real deployment's actual
// workload looks like: seed-compressed eval keys and query, reconstructed
// server-side, processed with real parallelism. Its purpose, like the
// non-seeded parallel test, is to catch anything the sequential seeded test
// couldn't -- a race condition, an aliasing bug, or a construction mistake
// specific to running this under threads.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   OMP_NUM_THREADS=16 ./test_full_scoring_with_splits_seeded_parallel

#include <gtest/gtest.h>

#include <omp.h>

#include <memory>
#include <random>
#include <vector>

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

TEST(FullScoringWithSplitsSeededParallel, MatchesResultUsingSeededMaterialUnderMultithreading) {
    Params params = Params::make_test_params();

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully.";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);
    ASSERT_GT(params.splits_per_cluster, 1)
        << "This test needs splits_per_cluster > 1 to exercise multiple splits; got "
        << "cluster_size=" << params.cluster_size << ", n=" << params.n
        << ", splits_per_cluster=" << params.splits_per_cluster
        << ". Increase database_size or decrease num_clusters/n.";

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single split's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 3;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- Eval keys: seeded, wire-compressed, reconstructed. NO secret
        // key involved in reconstruction. -------------------------------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- Query: sample embedding values, seed-compress, reconstruct
        // (sequential -- cheap, l is small). -----------------------------------
        std::vector<SignedValue> messages;
        std::vector<int64_t> embedding_values;
        messages.reserve(static_cast<size_t>(params.embedding_length));
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);
            embedding_values.push_back(m.reduced);
        }

        SeededQuery query_wire = build_seeded_query(ctx, secret, embedding_values, params.num_clusters,
                                                     params.desired_cluster_index);
        ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);

        // --- Switch reconstructed embedding ciphertexts to RLWE/eval form,
        // using the RECONSTRUCTED pub, IN PARALLEL. -----------------------------
        std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval(query.embedding_cts.size());
        #pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < static_cast<int64_t>(query.embedding_cts.size()); ++j) {
            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, query.embedding_cts[static_cast<size_t>(j)]);
            query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
        }

        // --- Switch reconstructed selector ciphertexts to RGSW, using the
        // RECONSTRUCTED pub, IN PARALLEL. ---------------------------------------
        std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(query.selector_cts.size());
        #pragma omp parallel for schedule(dynamic)
        for (int64_t c = 0; c < static_cast<int64_t>(query.selector_cts.size()); ++c) {
            RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]);
            rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
        }

        // --- Per (cluster, split): score, mask, IN PARALLEL. Same pattern as
        // the non-seeded parallel test: each thread writes its own [s][c]
        // slot, no shared mutable state touched inside the parallel region.
        // desired_cluster_raw_values is safe without locking for the same
        // reason as before -- desired_cluster_index is fixed, so exactly one
        // thread ever writes to desired_cluster_raw_values[s], for each s. -----
        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values(
            static_cast<size_t>(params.splits_per_cluster));

        // Built via resize(), not the vector(count, prototype) fill
        // constructor -- unique_ptr's copy constructor is deleted.
        std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
            static_cast<size_t>(params.splits_per_cluster));
        for (auto& row : masked) {
            row.resize(static_cast<size_t>(params.num_clusters));
        }

        int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

        // Each thread needs its own rng.
        std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
        for (auto& r : thread_rngs) r.seed(rng());

        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < total_pairs; ++idx) {
            int64_t c = idx / params.splits_per_cluster;
            int64_t s = idx % params.splits_per_cluster;
            std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

            std::vector<DatabasePolynomialEvalForm> db_split;
            db_split.reserve(static_cast<size_t>(params.embedding_length));
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                db_split.push_back(build_random_database_polynomial_eval_form(ctx, params, local_rng));
            }

            if (c == params.desired_cluster_index) {
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    desired_cluster_raw_values[static_cast<size_t>(s)].push_back(
                        db_split[static_cast<size_t>(j)].raw_values);
                }
            }

            RLWECT score = compute_split_score(ctx, query_eval, db_split);

            RLWECT masked_val(ctx.rlwe_param);
            rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);

            masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);
        }

        // --- Sequential reduction: sum over clusters, per split. -----------------
        std::vector<RLWECT> final_result;
        final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            final_result.emplace_back(ctx.rlwe_param);
        }
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            for (int64_t c = 0; c < params.num_clusters; ++c) {
                final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)],
                                                           *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
            }
        }

        // --- Decrypt and verify EACH split's result independently. ----------------
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            Vector decrypted = secret.rlwe_sk->decrypt_vector(final_result[static_cast<size_t>(s)], ctx.encoding);
            std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);
            const auto& raw_values_for_split = desired_cluster_raw_values[static_cast<size_t>(s)];

            for (int64_t i = 0; i < params.n; ++i) {
                int64_t expected = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    int64_t db_reduced_i_j = reduce_mod(
                        raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)],
                        params.plaintext_modulus);
                    expected = (expected + messages[static_cast<size_t>(j)].reduced * db_reduced_i_j) % params.plaintext_modulus;
                }
                EXPECT_EQ(decrypted[i], expected)
                    << "Mod-p mismatch on iteration " << iter << ", split " << s << ", coefficient " << i
                    << ": expected=" << expected << " got=" << decrypted[i];

                int64_t true_sum = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    true_sum += messages[static_cast<size_t>(j)].raw *
                                raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)];
                }
                int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
                EXPECT_EQ(decoded_signed, true_sum)
                    << "Overflow/corruption on iteration " << iter << ", split " << s << ", coefficient " << i
                    << ": true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                    << " (mod-p residue was " << decrypted[i] << ").";
            }
        }
    }
}
