// test_rgsw_multiplication.cpp
//
// Tests the cluster-selection mechanism (Steps 4/5/6 of the protocol, the
// unit-vector side rather than the embedding/scoring side): a bit (0 or 1)
// is gadget-encrypted as an LWE ciphertext, key-switched to an RGSW
// ciphertext, then multiplied against a random RLWE ciphertext. Following
// your reference sample:
//
//   LWEToRGSWKeySwitchKey lwe_to_rgsw_ksk(*lwe_sk, *gadget_sk);
//   LWEGadgetCT gadget_ct = lwe_gadget_sk.gadget_encrypt(bit);
//   RLWEGadgetCT rgsw_ct = lwe_to_rgsw_ksk.lwe_to_rlwe_key_switch(gadget_ct);
//   ...
//   RLWECT product(rlwe_param);
//   rgsw_ct.mul(product, some_rlwe_ct);
//
// RGSW(bit) * RLWE(message) should decrypt to `message` unchanged if
// bit == 1, or to the all-zero vector if bit == 0 — this is exactly the
// "selector" property Step 7 of the protocol relies on to zero out every
// cluster except the chosen one before summing.
//
// The RLWE ciphertext multiplied against is NOT a scalar-in-coefficient-0
// ciphertext like the LWE->RLWE switch produces elsewhere in this test
// suite — it's a genuine dense polynomial, encrypted directly via
// RLWESK::encode_and_encrypt(Vector, encoding), with n independent random
// coefficients from the same signed embedding range used everywhere else
// (see db_polynomial.hpp). This matches what the real "score" ciphertext
// being cluster-selected in Step 7 actually looks like.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rgsw_multiplication

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

TEST(RgswMultiplication, SelectorBitTimesRandomRlweCiphertext) {
    Params params = Params::make_test_params();

    // This test doesn't touch clusters or splits at all -- pin
    // database_size = n locally so splits_per_cluster stays 1, independent
    // of whatever "real" (multi-split) database_size
    // Params::make_test_params() uses for the rest of the protocol.
    params.database_size = params.n;
    params.derive_dependent_parameters();

    // Multiplying by a 0/1 selector bit can never grow a value's magnitude
    // (|bit * value| <= |value|), so the only precondition that matters here
    // is that individual embedding-range values themselves are already
    // representable without ambiguity — the same thing products_can_overflow
    // checks for a single product, which is a stronger (superset) condition.
    ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") don't leave individual embedding values safely "
        << "representable; fix that before this test can mean anything.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Step 4/5: gadget-encrypt the selector bit, key-switch to RGSW.
        int64_t bit = bit_dist(rng);
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        RLWEGadgetCT rgsw_ct = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);

        // --- Build a random, dense RLWE ciphertext to multiply against —
        // this stands in for the real "score" ciphertext from Step 6. -----
        std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
        std::vector<int64_t> reduced_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            SignedValue v = sample_signed_value(params, rng);
            raw_values[static_cast<size_t>(i)] = v.raw;
            reduced_values[static_cast<size_t>(i)] = v.reduced;
        }
        Vector message_vec(reduced_values, params.n, params.plaintext_modulus);
        RLWECT rlwe_ct = secret.rlwe_sk->encode_and_encrypt(message_vec, ctx.encoding);

        // --- Step 6/7: RGSW(bit) * RLWE(message). -------------------------
        RLWECT product_ct(ctx.rlwe_param);
        rgsw_ct.mul(product_ct, rlwe_ct);

        Vector decrypted = secret.rlwe_sk->decrypt_vector(product_ct, ctx.encoding);
        std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = (bit == 1) ? raw_values[static_cast<size_t>(i)] : 0;
            EXPECT_EQ(decoded_signed[static_cast<size_t>(i)], expected)
                << "Mismatch on iteration " << iter << ", coefficient " << i << ": bit=" << bit
                << " raw message value=" << raw_values[static_cast<size_t>(i)]
                << " expected=" << expected
                << " got=" << decoded_signed[static_cast<size_t>(i)]
                << " (mod-p residue was " << decrypted[i] << ").";
        }
    }
}