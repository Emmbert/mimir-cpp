// test_database_scoring_correctness.cpp
//
// Verifies ServerDatabase's real-data loading is correct END TO END: loads
// the fixed test database (Params::kTestDatabaseFilePath /
// make_test_database_params()), generates a random test query embedding,
// and computes the score TWO ways:
//   1. Plaintext ground truth: raw dot product of the query against each
//      embedding's raw_values -- no crypto involved at all.
//   2. Full protocol: encrypt the query (LWE), switch to RLWE, multiply
//      against the REAL loaded database's eval-form polynomials (via
//      ServerDatabase::build_split), decrypt.
// If ServerDatabase's coefficient loading/transpose logic -- or the
// eval-form/NTT conversion it goes through -- were wrong in any way, these
// two would disagree. This is a stronger check than
// test_real_database_loading.cpp, which only compares raw_values against
// file bytes directly: this exercises the full eval-form/NTT/multiply/
// decrypt path against real data, not just "are the bytes read correctly."
//
// Requires cpp_database_files/test_db_MSMarco_5100_l192_rho4_c2.mdb (+ its
// .meta.json sidecar) to exist at Params::kTestDatabaseFilePath. No
// environment variable needed -- the path and its matching Params are both
// defined together in params.hpp, specifically so they can't drift apart.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_database_scoring_correctness

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

TEST(DatabaseScoringCorrectness, ProtocolScoreMatchesPlaintextDotProductForRealData) {
    ASSERT_TRUE(std::filesystem::exists(Params::kTestDatabaseFilePath))
        << "Expected test database file at " << Params::kTestDatabaseFilePath
        << " -- see convert_clusters_to_database.py to generate it.";

    Params params = Params::make_test_database_params();
    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);
    ServerDatabase db = ServerDatabase::load_from_file(Params::kTestDatabaseFilePath);

    ASSERT_EQ(db.embedding_length(), params.embedding_length)
        << "Database file's embedding_length doesn't match make_test_database_params() -- "
        << "did the file get regenerated with different settings without updating the matching Params?";
    ASSERT_EQ(db.num_clusters(), params.num_clusters);
    EXPECT_NO_THROW(validate_value_range(db, params));

    std::mt19937_64 rng(std::random_device{}());

    for (int64_t cluster = 0; cluster < params.num_clusters; ++cluster) {
        ASSERT_EQ(db.splits_in_cluster(cluster, params), 1)
            << "This test assumes one split per cluster for simplicity -- cluster " << cluster
            << " has cluster_size=" << db.cluster_size(cluster) << ", needing "
            << db.splits_in_cluster(cluster, params) << " splits at n=" << params.n
            << ". Either the database changed, or n in make_test_database_params() needs raising.";

        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Random test query. -------------------------------------------
        std::vector<SignedValue> query;
        query.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            query.push_back(sample_signed_value(params, rng));
        }

        // --- Real loaded database, split 0. --------------------------------
        std::vector<DatabasePolynomialEvalForm> db_split = db.build_split(ctx, params, cluster, 0);
        ASSERT_EQ(static_cast<int64_t>(db_split.size()), params.embedding_length);

        // --- Path 1: full protocol -- encrypt, switch, multiply, decrypt. --
        std::vector<RLWECTEvalForm> query_eval;
        query_eval.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(query[static_cast<size_t>(j)].reduced, ctx.encoding);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
            query_eval.emplace_back(rlwe_ct);
        }
        RLWECT score = compute_split_score(ctx, query_eval, db_split);
        Vector decrypted = secret.rlwe_sk->decrypt_vector(score, ctx.encoding);
        std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

        // --- Path 2: plaintext ground truth, straight from raw_values. -----
        int64_t cluster_size = db.cluster_size(cluster);
        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                expected +=
                    query[static_cast<size_t>(j)].raw * db_split[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            }
            // Coefficients past cluster_size are zero-padding on both sides
            // (raw_values is already zero there, by build_split's contract),
            // so 'expected' naturally comes out 0 there too -- no special
            // casing needed, but the message below distinguishes the two
            // cases for easier debugging if this ever fails.
            EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                << "Mismatch at cluster " << cluster << ", coefficient " << i << " ("
                << (i < cluster_size ? "real embedding" : "zero-padding") << "): protocol="
                << decoded_signed[static_cast<size_t>(i)] << " plaintext=" << expected;
        }
    }
}
