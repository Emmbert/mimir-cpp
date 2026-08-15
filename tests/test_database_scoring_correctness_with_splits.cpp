// test_database_scoring_correctness_with_splits.cpp
//
// Same verification as test_database_scoring_correctness.cpp -- full
// protocol score vs plaintext ground truth, using the real test database --
// but with a smaller n (see Params::make_test_database_params_with_splits())
// so both of this file's clusters (2550 embeddings each) span MULTIPLE real
// splits: split 0 is fully real (2048 embeddings), split 1 is PARTIAL (502
// real + 1546 zero-padded). This is the real-data equivalent of
// test_full_scoring_with_splits.cpp (which uses random data) -- together
// with the single-split version, these two tests isolate two separate
// concerns: that one checks loading/NTT correctness in isolation, this one
// specifically exercises ServerDatabase's split-boundary/zero-padding logic
// and cross-split scoring against real data.
//
// Requires the same cpp_database_files/test_db_MSMarco_5100_l192_rho4_c2.mdb
// as test_database_scoring_correctness.cpp.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_database_scoring_correctness_with_splits

#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "server_db.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                            const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

} // namespace

TEST(DatabaseScoringCorrectnessWithSplits, ProtocolScoreMatchesPlaintextDotProductAcrossRealSplits) {
    ASSERT_TRUE(std::filesystem::exists(Params::kTestDatabaseFilePath))
        << "Expected test database file at " << Params::kTestDatabaseFilePath
        << " -- see convert_clusters_to_database.py to generate it.";

    Params params = Params::make_test_database_params_with_splits();
    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);
    ServerDatabase db = ServerDatabase::load_from_file(Params::kTestDatabaseFilePath);

    ASSERT_EQ(db.embedding_length(), params.embedding_length)
        << "Database file's embedding_length doesn't match the Params -- "
        << "did the file get regenerated with different settings?";
    ASSERT_EQ(db.num_clusters(), params.num_clusters);
    EXPECT_NO_THROW(validate_value_range(db, params));

    std::mt19937_64 rng(std::random_device{}());

    for (int64_t cluster = 0; cluster < params.num_clusters; ++cluster) {
        int64_t cluster_size = db.cluster_size(cluster);
        int64_t splits = db.splits_in_cluster(cluster, params);

        ASSERT_GT(splits, 1) << "Expected multiple splits for cluster " << cluster << " (cluster_size="
                              << cluster_size << ", n=" << params.n << ") -- got " << splits
                              << ". If the real database changed, this test's whole premise no longer "
                              << "holds at this n; adjust make_test_database_params_with_splits().";
        ASSERT_NE(cluster_size % params.n, 0)
            << "Expected cluster_size to NOT be an exact multiple of n, so the last split actually "
            << "exercises zero-padding -- got an exact multiple, so this test wouldn't be testing what "
            << "it's meant to.";

        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Query: built once, reused across every split -- same as
        // test_full_scoring_with_splits.cpp's reasoning (a cluster's
        // selection doesn't depend on which split is being scored). --------
        std::vector<SignedValue> query;
        query.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            query.push_back(sample_signed_value(params, rng));
        }

        std::vector<RLWECTEvalForm> query_eval;
        query_eval.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(query[static_cast<size_t>(j)].reduced, ctx.encoding);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
            query_eval.emplace_back(rlwe_ct);
        }

        // --- Score and verify EACH split independently. ---------------------
        for (int64_t split = 0; split < splits; ++split) {
            std::vector<DatabasePolynomialEvalForm> db_split = db.build_split(ctx, params, cluster, split);
            ASSERT_EQ(static_cast<int64_t>(db_split.size()), params.embedding_length);

            RLWECT score = compute_split_score(ctx, query_eval, db_split);
            Vector decrypted = secret.rlwe_sk->decrypt_vector(score, ctx.encoding);
            std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

            int64_t split_start = split * params.n;
            for (int64_t i = 0; i < params.n; ++i) {
                int64_t expected = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    expected += query[static_cast<size_t>(j)].raw *
                                db_split[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
                }

                int64_t global_embedding_idx = split_start + i;
                EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                    << "Mismatch at cluster " << cluster << ", split " << split << ", coefficient " << i
                    << " (global embedding index " << global_embedding_idx << ", "
                    << (global_embedding_idx < cluster_size ? "real embedding" : "zero-padding")
                    << "): protocol=" << decoded_signed[static_cast<size_t>(i)] << " plaintext=" << expected;
            }
        }
    }
}
