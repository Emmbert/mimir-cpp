// test_process_query_correctness.cpp
//
// Directly exercises server_query_processing.cpp's process_query -- the
// ACTUAL function server.cpp calls for every real request -- against the
// real test database, comparing its decrypted result to an independently
// computed plaintext dot product.
//
// Every earlier scoring test (test_full_scoring_with_splits_seeded_parallel.cpp,
// benchmark_latency_seeded_parallel.cpp, test_database_scoring_correctness.cpp)
// inlines its OWN copy of the switch/score/mask/sum logic rather than
// calling process_query itself -- so a bug specific to process_query's own
// implementation (as opposed to the general pattern those other tests
// exercise) would not be caught by any of them. This test closes that gap.
//
// Also exercises the full seed-based wire path realistically: builds a
// SeededQuery via build_seeded_query (client side) and passes the
// WIRE-COMPRESSED form -- exactly what arrives over the network -- into
// process_query, rather than a pre-reconstructed ReconstructedQuery. This
// validates process_query's internal unpacking (reconstruct_query) in the
// exact context it's actually called from, not just standalone.
//
// Requires cpp_database_files/test_db_MSMarco_5100_l192_rho4_c2.mdb, same
// as the other database_scoring_correctness tests -- but note that file's
// embedding_length/precision (192/4) don't match make_test_database_params()
// (10/2); this test uses whichever Params factory actually matches whatever
// database file you point Params::kTestDatabaseFilePath at. Skips (rather
// than fails) if that file isn't present.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_process_query_correctness

#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"
#include "server_db.hpp"
#include "server_query_processing.hpp"

using namespace FHEDeck;
using namespace psearch;

TEST(ProcessQueryCorrectness, MatchesPlaintextDotProductForRealData) {
    if (!std::filesystem::exists(Params::kTestDatabaseFilePath)) {
        GTEST_SKIP() << "Expected test database file at " << Params::kTestDatabaseFilePath
                     << " -- see convert_clusters_to_database.py to generate it.";
    }

    Params params = Params::make_test_database_params();
    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);
    ServerDatabase db = ServerDatabase::load_from_file(Params::kTestDatabaseFilePath);

    ASSERT_EQ(db.embedding_length(), params.embedding_length)
        << "Database file doesn't match make_test_database_params() -- point "
        << "Params::kTestDatabaseFilePath at a file with matching embedding_length/precision.";
    ASSERT_EQ(db.num_clusters(), params.num_clusters);
    ASSERT_NO_THROW(validate_value_range(db, params));
    ASSERT_NO_THROW(validate_uniform_cluster_sizes(db, params));

    std::mt19937_64 rng(std::random_device{}());

    // --- Eval keys: seeded, reconstructed -- exactly what server.cpp does
    // for a Registration message. ------------------------------------------
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
    ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

    // --- Query: seeded -- exactly the wire-compressed form that would
    // actually arrive over the network for a Query message. -------------------
    std::vector<SignedValue> query_values;
    std::vector<int64_t> embedding_values;
    query_values.reserve(static_cast<size_t>(params.embedding_length));
    embedding_values.reserve(static_cast<size_t>(params.embedding_length));
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        SignedValue v = sample_signed_value(params, rng);
        query_values.push_back(v);
        embedding_values.push_back(v.reduced);
    }
    SeededQuery query_wire = build_seeded_query(ctx, secret, embedding_values, params.num_clusters,
                                                 params.desired_cluster_index);

    // --- Call process_query DIRECTLY -- exactly what server.cpp does. ---------
    QueryResponse response = process_query(ctx, params, db, pub, query_wire);

    ASSERT_EQ(static_cast<int64_t>(response.ciphertexts.size()), params.splits_per_cluster);

    // --- Independently computed plaintext ground truth, for the desired
    // cluster (the only one whose contribution survives RGSW masking). --------
    for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
        Vector decrypted = secret.rlwe_sk->decrypt_vector(response.ciphertexts[static_cast<size_t>(s)], ctx.encoding);
        std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

        std::vector<DatabasePolynomialEvalForm> desired_split =
            db.build_split(ctx, params, params.desired_cluster_index, s);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                expected += query_values[static_cast<size_t>(j)].raw *
                            desired_split[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            }
            EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                << "Mismatch at split " << s << ", coefficient " << i << ": process_query="
                << decoded_signed[static_cast<size_t>(i)] << " plaintext=" << expected;
        }
    }
}
