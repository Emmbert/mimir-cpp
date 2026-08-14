// test_seeded_key_material_and_query_roundtrip.cpp
//
// The full pipeline in one test: eval keys are built with a seed (client
// side), only the wire-compressed form (SeededClientPublicMaterial) is kept,
// and reconstructed into a real, usable ClientPublicMaterial with NO secret
// key involved (server side) -- see seeded_eval_keys.hpp. Separately, several
// LWE ciphertexts are built with a second, independent seed, and only
// (seed, b) is kept for each -- see seeded_distribution.hpp and
// test_seeded_lwe_to_rlwe_roundtrip.cpp for that half of the pattern.
//
// Two checks, using the RECONSTRUCTED public material throughout (never the
// original pub built alongside the wire structs):
//   1. LWE->RLWE: reconstructed LWE ciphertexts, switched via the
//      reconstructed lwe_to_rlwe_ksk, decrypted -- exercises
//      LWEToRLWEKeySwitchKey's reconstruction constructor.
//   2. LWE'->RGSW: a cluster-selector-style ciphertext, switched via the
//      reconstructed lwe_to_rgsw_ksk, then RGSW-multiplied against a random
//      dense RLWE ciphertext, decrypted -- exercises ct_of_sk_dest's
//      reconstruction specifically, which check 1 never touches.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_seeded_key_material_and_query_roundtrip

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"

using namespace FHEDeck;
using namespace psearch;


TEST(SeededKeyMaterialAndQueryRoundtrip, ReconstructedEvalKeysSwitchAndDecryptCorrectly) {
    Params params = Params::make_test_params();
    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 5;
    constexpr int kNumCiphertextsPerIteration = 6; // "several" LWE ciphertexts per iteration

    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- 1. Eval keys: seeded generation, wire-compressed, then
        // reconstructed. ------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- 2. Several LWE (embedding-style) ciphertexts: seeded, kept as
        // (seed, b) only, then reconstructed. --------------------------------
        auto query_seed = generate_fresh_seed();
        auto query_dist = std::make_shared<SeededUniformDistribution>(query_seed, 0, params.q);
        std::shared_ptr<Distribution> original_lwe_dist = secret.lwe_sk->set_unif_dist(query_dist);

        std::vector<int64_t> messages;
        std::vector<int64_t> b_values;
        messages.reserve(kNumCiphertextsPerIteration);
        b_values.reserve(kNumCiphertextsPerIteration);

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            int64_t m = sample_signed_mod_value(params, rng);
            messages.push_back(m);

            LWECT ct = secret.lwe_sk->encode_and_encrypt(m, ctx.encoding);
            b_values.push_back(ct[0]); // this + query_seed is "all that got sent"
        }

        secret.lwe_sk->set_unif_dist(original_lwe_dist); // restore normal behaviour

        // --- Check 1: reconstruct each ciphertext, switch via the
        // RECONSTRUCTED lwe_to_rlwe_ksk, decrypt normally. --------------------
        auto reconstruct_dist = std::make_shared<SeededUniformDistribution>(query_seed, 0, params.q);

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            LWECT reconstructed =
                reconstruct_lwe_from_seed_and_b(secret.lwe_sk->param(), *reconstruct_dist, b_values[k]);

            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, reconstructed);

            Vector decrypted_vector = secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.encoding);
            int64_t decrypted = static_cast<int64_t>(decrypted_vector[0]);

            EXPECT_EQ(decrypted, messages[static_cast<size_t>(k)])
                << "LWE->RLWE mismatch on iteration " << iter << ", ciphertext " << k << ": encrypted "
                << messages[static_cast<size_t>(k)] << " but decrypted to " << decrypted;
        }

        // --- Check 2: a selector-style LWE' ciphertext, switched via the
        // RECONSTRUCTED lwe_to_rgsw_ksk (exercises ct_of_sk_dest's
        // reconstruction specifically), then RGSW-multiplied against a
        // random dense RLWE ciphertext and decrypted. --------------------------
        int64_t bit = 1; // arbitrary; correctness doesn't depend on which value
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        RLWEGadgetCT rgsw_ct = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);

        std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
        std::vector<int64_t> reduced_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            SignedValue v = sample_signed_value(params, rng);
            raw_values[static_cast<size_t>(i)] = v.raw;
            reduced_values[static_cast<size_t>(i)] = v.reduced;
        }
        Vector message_vec(reduced_values, params.n, params.plaintext_modulus);
        RLWECT rlwe_ct = secret.rlwe_sk->encode_and_encrypt(message_vec, ctx.encoding);

        RLWECT product_ct(ctx.rlwe_param);
        rgsw_ct.mul(product_ct, rlwe_ct);

        Vector decrypted = secret.rlwe_sk->decrypt_vector(product_ct, ctx.encoding);
        std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = (bit == 1) ? raw_values[static_cast<size_t>(i)] : 0;
            EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                << "LWE'->RGSW mismatch on iteration " << iter << ", coefficient " << i
                << ": expected=" << expected << " got=" << decoded_signed[static_cast<size_t>(i)];
        }
    }
}
