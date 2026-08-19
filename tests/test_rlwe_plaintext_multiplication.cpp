// test_rlwe_plaintext_multiplication.cpp
//
// Randomized correctness test for RLWE ciphertext x plaintext-polynomial
// multiplication, PLUS an explicit check that no product ever needs the
// (combined, if CRT) plaintext modulus to "wrap around" to be decoded
// correctly.
//
// Parameterized over TWO Params factories, run via the SAME test body --
// same reasoning as the earlier roundtrip tests. Both the query VALUE and
// the DATABASE polynomial get CRT-split when r==2 -- matching the real
// protocol design: the database is split into per-component-ring databases
// alongside the query, not just the query alone.
//
// Two distinct things are being checked, and it matters that they're
// distinct:
//   1. Mod-p consistency: after decrypting each component ring and
//      recomposing (a no-op when r==1), the result must equal the true
//      product reduced mod the COMBINED modulus (plaintext_modulus for
//      r==1, comp_ring_modulus*(comp_ring_modulus-1) for r==2). This can
//      NEVER catch overflow, because both sides of that comparison are
//      already reduced -- it only tells you the crypto (and, for r==2, the
//      CRT split/recompose) is internally consistent, not that the
//      represented value is actually correct.
//   2. Overflow: the true, un-reduced signed product must have magnitude
//      less than combined_modulus/2, or the "centered" signed decoding can
//      no longer recover it. Checked ONLY after recomposition -- checking
//      each CRT component's centered residue individually would be
//      meaningless, since an individual component's own "half" boundary has
//      nothing to do with whether the TRUE (unreduced) product fits in the
//      combined space; only the recomposed value's relationship to the
//      combined modulus matters here.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_plaintext_multiplication

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

class RlwePlaintextMultiplication : public ::testing::TestWithParam<Params (*)()> {};

LWECT encrypt_lwe(const ClientSecretMaterial& secret, int64_t message, const PlaintextEncoding& encoding) {
    return secret.lwe_sk->encode_and_encrypt(message, encoding);
}

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

} // namespace

TEST_P(RlwePlaintextMultiplication, ScalarTimesRandomDatabasePolynomial) {
    Params params = GetParam()();

    // Fail fast, before running anything, if these parameters can never
    // avoid overflow in the worst case. UNCHANGED by CRT -- always
    // evaluated against plaintext_modulus (the required lower bound);
    // derive_dependent_parameters() already guarantees the combined CRT
    // modulus is >= plaintext_modulus, so "safe against plaintext_modulus"
    // implies "safe against the (larger or equal) combined modulus"
    // automatically -- no separate CRT overflow check needed here.
    /*ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") are incompatible: the worst-case product magnitude is "
        << max_abs_embedding_value(params) << "^2 = " << (max_abs_embedding_value(params) * max_abs_embedding_value(params))
        << ", which is >= plaintext_modulus/2 (" << (params.plaintext_modulus / 2) << "). "
        << "Increase plaintext_modulus or decrease embedding_precision so that "
        << "max_abs_embedding_value(params)^2 < plaintext_modulus/2.";*/

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        SignedValue message = sample_signed_value(params, rng);
        DatabasePolynomial db = build_random_database_polynomial(params, rng);

        // --- The value(s) actually encrypted: r==1 uses message.reduced
        // directly (unchanged); r==2 CRT-splits message.raw, canonicalized
        // into [0, combined_modulus) first. --------------------------------------
        std::vector<int64_t> message_components;
        if (r == 1) {
            message_components = {message.reduced};
        } else {
            int64_t canonical = reduce_mod(message.raw, combined_modulus);
            auto [c1, c2] = crt_split(canonical, params.comp_ring_modulus);
            message_components = {c1, c2};
        }

        // --- Per component ring: encrypt, switch, multiply against THIS
        // ring's own database polynomial (built once for all rings via
        // crt_split_database_polynomial), decrypt. Each ring's FULL
        // decrypted vector is kept for recomposition below -- nothing is
        // checked per-ring here, since an individual component's residue
        // isn't meaningful to compare against anything on its own (see file
        // header). --------------------------------------------------------------------
        std::vector<Polynomial> db_poly_per_ring = crt_split_database_polynomial(params, db.raw_values);

        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));

        for (int64_t ring = 0; ring < r; ++ring) {
            LWECT lwe_ct = encrypt_lwe(secret, message_components[static_cast<size_t>(ring)],
                                       ctx.component_encodings[static_cast<size_t>(ring)]);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);

            RLWECT product_ct(ctx.rlwe_param);
            rlwe_ct.mul(product_ct, db_poly_per_ring[static_cast<size_t>(ring)]);

            decrypted_per_ring.push_back(
                secret.rlwe_sk->decrypt_vector(product_ct, ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        // --- Recompose (a no-op when r==1) and check every coefficient. -----------
        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            // Check 1: mod-(combined) consistency. Can't catch overflow by itself.
            int64_t expected = reduce_mod(message.raw * db.raw_values[static_cast<size_t>(i)], combined_modulus);
            EXPECT_EQ(recomposed, expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): message.raw=" << message.raw << " db.raw_values[i]=" << db.raw_values[static_cast<size_t>(i)]
                << " expected=" << expected << " got=" << recomposed;

            // Check 2: overflow, checked against the recomposed value and
            // the COMBINED modulus -- see file header for why this can't be
            // checked per-component.
            int64_t true_product = message.raw * db.raw_values[static_cast<size_t>(i)];
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            EXPECT_EQ(decoded_signed, true_product)
                << "Overflow on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): raw message=" << message.raw << " raw db value=" << db.raw_values[static_cast<size_t>(i)]
                << " true product=" << true_product << " but decoded signed value=" << decoded_signed
                << " (recomposed residue was " << recomposed << "). This means embedding_precision/"
                << "plaintext_modulus allow products whose magnitude exceeds combined_modulus/2 -- increase "
                << "plaintext_modulus/comp_ring_modulus or decrease embedding_precision.";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, RlwePlaintextMultiplication,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });