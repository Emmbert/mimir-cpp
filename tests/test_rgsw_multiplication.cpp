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
// bit == 1, or to the all-zero vector if bit == 0 -- this is exactly the
// "selector" property Step 7 of the protocol relies on to zero out every
// cluster except the chosen one before summing.
//
// Parameterized over TWO Params factories, run via the SAME test body.
// The selector itself is UNCHANGED by CRT -- confirmed early on: exactly
// ONE unit-vector encryption regardless of num_component_rings, since the
// selector is never CRT-split. What DOES change is the "message" being
// multiplied against: it stands in for a real score, and scores come from
// CRT-split databases (see test_rlwe_scoring_dot_product.cpp), so it gets
// CRT-split into per-ring Vectors here -- multiplied by the SAME rgsw_ct
// for every ring (matching the real protocol: "the same ciphertext is
// multiplied with the scoring results of both CRT databases").
//
// Only ONE check here, not the two-check split used in the multiplication/
// scoring tests: multiplying by a 0/1 selector bit can never grow a value's
// magnitude (|bit * value| <= |value|), so there's no overflow risk to
// distinguish from mod-consistency -- a single "centered residue of the
// recomposed value equals the true (possibly negative) expected value"
// check is both necessary and sufficient, matching the original non-CRT
// test's single check exactly. Note this MUST use centered_residue on the
// recomposed value, not a direct equality check -- crt_recompose always
// returns a non-negative canonical residue in [0, combined_modulus), while
// `expected` can be negative (raw_values[i] is a signed embedding value);
// comparing them directly would be wrong.
//
// The RLWE ciphertext multiplied against is NOT a scalar-in-coefficient-0
// ciphertext like the LWE->RLWE switch produces elsewhere in this test
// suite -- it's a genuine dense polynomial, encrypted directly via
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

#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class RgswMultiplication : public ::testing::TestWithParam<Params (*)()> {};

} // namespace

TEST_P(RgswMultiplication, SelectorBitTimesRandomRlweCiphertext) {
    Params params = GetParam()();

    // This test doesn't touch clusters or splits at all -- pin
    // database_size = n locally so splits_per_cluster stays 1, independent
    // of whatever "real" (multi-split) database_size the factory uses for
    // the rest of the protocol.
    params.database_size = params.n;
    params.derive_dependent_parameters();

    // Multiplying by a 0/1 selector bit can never grow a value's magnitude
    // (|bit * value| <= |value|), so the only precondition that matters here
    // is that individual embedding-range values themselves are already
    // representable without ambiguity -- the same thing products_can_overflow
    // checks for a single product, which is a stronger (superset) condition.
    // UNCHANGED by CRT -- always evaluated against plaintext_modulus (the
    // required lower bound), which derive_dependent_parameters() already
    // guarantees the combined CRT modulus meets or exceeds.
    ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") don't leave individual embedding values safely "
        << "representable; fix that before this test can mean anything.";

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Step 4/5: gadget-encrypt the selector bit, key-switch to RGSW.
        // UNCHANGED by CRT -- exactly ONE selector, reused for every
        // component ring below. --------------------------------------------------------
        int64_t bit = bit_dist(rng);
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        RLWEGadgetCT rgsw_ct = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);

        // --- Build a random, dense "score" -- raw values sampled once,
        // then CRT-split into per-ring Vectors below. This stands in for the
        // real "score" ciphertext from Step 6. -----------------------------------------
        std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            raw_values[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
        }

        // --- Per component ring: encrypt this ring's own reduced Vector,
        // multiply by the SAME rgsw_ct, decrypt. --------------------------------------
        std::vector<Vector> message_vecs = crt_split_vector_from_raw_values(params, raw_values);

        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));

        for (int64_t ring = 0; ring < r; ++ring) {
            RLWECT rlwe_ct = secret.rlwe_sk->encode_and_encrypt(message_vecs[static_cast<size_t>(ring)],
                                                                 ctx.component_encodings[static_cast<size_t>(ring)]);

            // --- Step 6/7: RGSW(bit) * RLWE(message), SAME rgsw_ct for every ring. ---
            RLWECT product_ct(ctx.rlwe_param);
            rgsw_ct.mul(product_ct, rlwe_ct);

            decrypted_per_ring.push_back(
                secret.rlwe_sk->decrypt_vector(product_ct, ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        // --- Recompose (a no-op when r==1) and check every coefficient. -----------
        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            // centered_residue is required here, not a direct comparison:
            // recomposed is always non-negative in [0, combined_modulus),
            // but expected can be negative -- see file header.
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            int64_t expected = (bit == 1) ? raw_values[static_cast<size_t>(i)] : 0;

            EXPECT_EQ(decoded_signed, expected)
                << "Mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r << "): bit=" << bit
                << " raw message value=" << raw_values[static_cast<size_t>(i)] << " expected=" << expected
                << " got=" << decoded_signed << " (recomposed residue was " << recomposed << ").";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, RgswMultiplication,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });
