// test_full_scoring_with_splits_parallel.cpp
//
// Same protocol and same correctness checks as
// test_full_scoring_with_splits.cpp, but RLWE switching, RGSW switching, and
// the per-(cluster,split) scoring are parallelized via OpenMP -- the exact
// same threading pattern used in benchmarks/benchmark_latency_parallel.cpp.
//
// This is a correctness sanity check for that parallelization, not a new
// protocol test: the expected results are identical to the single-threaded
// version, computed with multiple threads. Its whole purpose is to catch
// anything the single-threaded tests structurally couldn't -- a race
// condition on shared state, an aliasing bug, or a construction mistake like
// the vector<vector<unique_ptr<T>>> fill-constructor issue that showed up
// while building the benchmark.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   OMP_NUM_THREADS=16 ./test_full_scoring_with_splits_parallel

#include <gtest/gtest.h>

#include <omp.h>

#include <memory>
#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

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

TEST(FullScoringWithSplitsParallel, MatchesSingleThreadedResultUnderMultithreading) {
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
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Query: sample + encrypt sequentially (cheap, l is small), then
        // key-switch to RLWE and convert to eval form IN PARALLEL. ---------
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<LWECT> embedding_lwe;
        embedding_lwe.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages[static_cast<size_t>(j)] = m;
            embedding_lwe.push_back(secret.lwe_sk->encode_and_encrypt(m.reduced, ctx.encoding));
        }

        std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval(static_cast<size_t>(params.embedding_length));
        #pragma omp parallel for schedule(dynamic)
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, embedding_lwe[static_cast<size_t>(j)]);
            query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
        }

        // --- Cluster-selection: gadget-encrypt sequentially, key-switch to
        // RGSW IN PARALLEL. --------------------------------------------------
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

        // --- Per (cluster, split): score, mask, IN PARALLEL. Each thread
        // writes its result into its own [s][c] slot -- no shared mutable
        // state touched inside the parallel region. desired_cluster_raw_values
        // is also safe to fill here without locking: desired_cluster_index is
        // fixed, so exactly one thread ever writes to
        // desired_cluster_raw_values[s], for each s. ------------------------
        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values(
            static_cast<size_t>(params.splits_per_cluster));

        // Built via resize(), not the vector(count, prototype) fill
        // constructor -- that needs to COPY the prototype inner vector,
        // and unique_ptr's copy constructor is deleted.
        std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
            static_cast<size_t>(params.splits_per_cluster));
        for (auto& row : masked) {
            row.resize(static_cast<size_t>(params.num_clusters));
        }

        int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

        // Each thread needs its own rng -- std::mt19937_64 isn't thread-safe
        // to share. Seed each thread's rng once, sequentially, before the
        // parallel region.
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

        // --- Sequential reduction: sum over clusters, per split. Cheap
        // relative to the parallel work above -- see benchmark_latency_parallel.cpp
        // for why this isn't also parallelized. -----------------------------
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

        // --- Decrypt and verify EACH split's result independently -- same
        // checks as the single-threaded test. --------------------------------
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
