// test_seeded_full_query_roundtrip.cpp
//
// The complete picture: eval keys AND the query are both built with seeds,
// wire-compressed, and reconstructed server-side with NO secret key
// involved in reconstruction at any point -- see seeded_eval_keys.hpp and
// seeded_query.hpp. This is the closest thing so far to what a real
// deployment's communication pattern would actually look like.
//
// Two checks, using ONLY reconstructed material throughout:
//   1. Each reconstructed embedding ciphertext, switched via the
//      reconstructed lwe_to_rlwe_ksk, decrypted and compared to the
//      original plaintext value.
//   2. Each reconstructed selector ciphertext, switched via the
//      reconstructed lwe_to_rgsw_ksk, RGSW-multiplied against a random
//      dense RLWE ciphertext, decrypted -- verifying the selector bit's
//      value (1 at desired_cluster_index, 0 elsewhere) came through
//      correctly end-to-end.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_seeded_full_query_roundtrip

#include <gtest/gtest.h>

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

TEST(SeededFullQueryRoundtrip, EverythingSeededAndReconstructedEndToEnd) {
    Params params = Params::make_test_params();
    CryptoContext ctx = CryptoContext::from_params(params);

    ASSERT_GT(params.num_clusters, 0);
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);

    constexpr int kNumIterations = 3; // each iteration does a full eval-key build +
                                       // reconstruction, plus num_clusters selector
                                       // switches -- keep modest.

    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- Eval keys: seeded, wire-compressed, reconstructed. --------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- Query: seeded, wire-compressed, reconstructed. -------------------
        std::vector<int64_t> embedding_values;
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            embedding_values.push_back(sample_signed_mod_value(params, rng));
        }

        SeededQuery query_wire = build_seeded_query(ctx, secret, embedding_values, params.num_clusters,
                                                     params.desired_cluster_index);
        ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);

        // --- Check 1: embedding ciphertexts, switched via reconstructed
        // lwe_to_rlwe_ksk, decrypted. -------------------------------------------
        ASSERT_EQ(query.embedding_cts.size(), embedding_values.size());
        for (size_t j = 0; j < query.embedding_cts.size(); ++j) {
            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, query.embedding_cts[j]);

            Vector decrypted_vector = secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.encoding);
            int64_t decrypted = static_cast<int64_t>(decrypted_vector[0]);

            EXPECT_EQ(decrypted, embedding_values[j])
                << "Embedding mismatch on iteration " << iter << ", index " << j << ": encrypted "
                << embedding_values[j] << " but decrypted to " << decrypted;
        }

        // --- Check 2: selector ciphertexts, switched via reconstructed
        // lwe_to_rgsw_ksk, RGSW-multiplied, decrypted. ---------------------------
        std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
        std::vector<int64_t> reduced_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            SignedValue v = sample_signed_value(params, rng);
            raw_values[static_cast<size_t>(i)] = v.raw;
            reduced_values[static_cast<size_t>(i)] = v.reduced;
        }
        Vector message_vec(reduced_values, params.n, params.plaintext_modulus);
        RLWECT rlwe_ct = secret.rlwe_sk->encode_and_encrypt(message_vec, ctx.encoding);

        ASSERT_EQ(query.selector_cts.size(), static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            RLWEGadgetCT rgsw_ct = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]);

            RLWECT product_ct(ctx.rlwe_param);
            rgsw_ct.mul(product_ct, rlwe_ct);

            Vector decrypted = secret.rlwe_sk->decrypt_vector(product_ct, ctx.encoding);
            std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

            int64_t expected_bit = (c == params.desired_cluster_index) ? 1 : 0;
            for (int64_t i = 0; i < params.n; ++i) {
                int64_t expected = (expected_bit == 1) ? raw_values[static_cast<size_t>(i)] : 0;
                EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                    << "Selector mismatch on iteration " << iter << ", cluster " << c << ", coefficient " << i
                    << ": expected=" << expected << " got=" << decoded_signed[static_cast<size_t>(i)];
            }
        }
    }
}
