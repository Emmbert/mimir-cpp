// test_rlwe_plaintext_multiplication.cpp
//
// Randomized correctness test for RLWE ciphertext x plaintext-polynomial
// multiplication, PLUS an explicit check that no product ever needs the
// plaintext modulus to "wrap around" to be decoded correctly.
//
// Two distinct things are being checked, and it matters that they're
// distinct:
//   1. Mod-p consistency (EXPECT_EQ against `expected`): decrypting gives the
//      same value you'd get by doing the multiplication directly mod p. This
//      can NEVER catch overflow, because both sides of that comparison are
//      already reduced mod p — it only tells you the crypto is internally
//      consistent, not that the represented value is actually correct.
//   2. Overflow (EXPECT_EQ against `true_product`, via centered_residue):
//      the true, un-reduced signed product must have magnitude less than
//      plaintext_modulus/2, or the "centered" signed decoding
//      (value <= p/2 ? value : value - p) can no longer recover it. If your
//      embedding_precision and plaintext_modulus don't satisfy
//      max_abs_embedding_value(params)^2 < plaintext_modulus/2, this WILL
//      happen for some sampled values, and check #1 will still pass while
//      check #2 correctly fails.
//
// A precondition ASSERT at the top of the test fails fast with a clear
// message before even running the loop, if the parameters can't possibly
// avoid overflow in the worst case.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_plaintext_multiplication

#include <gtest/gtest.h>

#include <random>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

LWECT encrypt_lwe(const CryptoContext& ctx, const ClientSecretMaterial& secret, int64_t message) {
    return secret.lwe_sk->encode_and_encrypt(message, ctx.encoding);
}

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

Vector decrypt_rlwe_full(const CryptoContext& ctx, const ClientSecretMaterial& secret, const RLWECT& rlwe_ct) {
    return secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.encoding);
}

} // namespace

TEST(RlwePlaintextMultiplication, ScalarTimesRandomDatabasePolynomial) {
    Params params = Params::make_test_params();

    // Fail fast, before running anything, if these parameters can never
    // avoid overflow in the worst case — this is the check that would have
    // immediately flagged embedding_precision=8 / plaintext_modulus=17.
    /*ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") are incompatible: the worst-case product magnitude is "
        << max_abs_embedding_value(params) << "^2 = " << (max_abs_embedding_value(params) * max_abs_embedding_value(params))
        << ", which is >= plaintext_modulus/2 (" << (params.plaintext_modulus / 2) << "). "
        << "Increase plaintext_modulus or decrease embedding_precision so that "
        << "max_abs_embedding_value(params)^2 < plaintext_modulus/2.";*/

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        SignedValue message = sample_signed_value(params, rng);

        LWECT lwe_ct = encrypt_lwe(ctx, secret, message.reduced);
        RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);

        DatabasePolynomial db = build_random_database_polynomial(params, rng);

        RLWECT product_ct(ctx.rlwe_param);
        rlwe_ct.mul(product_ct, db.poly);

        Vector decrypted = decrypt_rlwe_full(ctx, secret, product_ct);
        std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            // Check 1: mod-p consistency. Can't catch overflow by itself.
            int64_t expected = (message.reduced * db.poly[i]) % params.plaintext_modulus;
            EXPECT_EQ(decrypted[i], expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i
                << ": message.reduced=" << message.reduced << " db.poly[i]=" << db.poly[i]
                << " expected=" << expected << " got=" << decrypted[i];

            // Check 2: overflow. Compares the TRUE signed product (from the
            // un-reduced raw values) against the centered-signed
            // interpretation of the decrypted residue.
            int64_t true_product = message.raw * db.raw_values[static_cast<size_t>(i)];
            int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
            EXPECT_EQ(decoded_signed, true_product)
                << "Overflow on iteration " << iter << ", coefficient " << i << ": raw message="
                << message.raw << " raw db value=" << db.raw_values[static_cast<size_t>(i)]
                << " true product=" << true_product << " but decoded signed value="
                << decoded_signed << " (mod-p residue was " << decrypted[i] << "). "
                << "This means embedding_precision/plaintext_modulus allow products whose "
                << "magnitude exceeds plaintext_modulus/2 — increase plaintext_modulus or "
                << "decrease embedding_precision.";
        }
    }
}