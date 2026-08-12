// test_rlwe_plaintext_multiplication_ntt.cpp
//
// Same test as test_rlwe_plaintext_multiplication.cpp (same sampling, same
// mod-p consistency check, same overflow check), but performs the actual
// ciphertext x plaintext-polynomial multiplication in NTT/eval form instead
// of coefficient form.
//
// The database polynomial is built directly in eval/NTT form via
// build_random_database_polynomial_eval_form (db_polynomial.hpp) — there is
// no coefficient-form Polynomial in this file at all for the database side;
// that intermediate lives and dies entirely inside that builder function.
// This matches how the real database will eventually be stored (Step 1 of
// the protocol: "All polynomials are transferred into NTT representation") —
// ServerDatabase::build will call the same builder for every polynomial in
// every split of every cluster, so the database is in eval form from the
// moment it's created, never converted at query time.
//
// The ciphertext side still needs an explicit conversion, following the
// reference perf_test_sequential() sample:
//
//   RLWECTEvalForm ct_eval(ct);              // ciphertext coef-form -> eval-form
//   RLWECTEvalForm product(rlwe_param);
//   ct_eval.mul(product, *database_entry_eval_form);   // multiply in eval form
//   RLWECT out(product);                     // eval-form -> coef-form, for decrypt
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_plaintext_multiplication_ntt

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

TEST(RlwePlaintextMultiplicationNtt, ScalarTimesRandomDatabasePolynomialEvalForm) {
    Params params = Params::make_test_params();

    // Same precondition as the coefficient-form version — overflow is a
    // property of the parameters/values, not of which multiplication path
    // is used, so this check doesn't change.
    ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") are incompatible: the worst-case product magnitude is "
        << max_abs_embedding_value(params) << "^2 = "
        << (max_abs_embedding_value(params) * max_abs_embedding_value(params))
        << ", which is >= plaintext_modulus/2 (" << (params.plaintext_modulus / 2) << "). "
        << "Increase plaintext_modulus or decrease embedding_precision so that "
        << "max_abs_embedding_value(params)^2 < plaintext_modulus/2.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        SignedValue message = sample_signed_value(params, rng);

        LWECT lwe_ct = encrypt_lwe(ctx, secret, message.reduced);
        RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);

        // Database polynomial is built directly in eval/NTT form — the
        // coefficient-form intermediate never leaves build_random_database_
        // polynomial_eval_form; this matches how the real database will
        // eventually be stored (see db_polynomial.hpp).
        DatabasePolynomialEvalForm db = build_random_database_polynomial_eval_form(ctx, params, rng);

        // Ciphertext: coefficient form -> eval form, via the conversion
        // constructor (same as `RLWECTEvalForm(ct)` in the reference sample).
        RLWECTEvalForm rlwe_ct_eval(rlwe_ct);

        // Multiply directly in eval form.
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        rlwe_ct_eval.mul(product_eval, *db.poly_eval);

        // Back to coefficient form for decryption, via the conversion
        // constructor (same as `RLWECT(out_ct)` in the reference sample).
        RLWECT product_ct(product_eval);

        // --- everything below is identical to the coefficient-form test ---

        Vector decrypted = decrypt_rlwe_full(ctx, secret, product_ct);
        std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            // Check 1: mod-p consistency. db.poly_eval has no coefficient-form
            // indexing (it's already in eval/NTT form), so recompute the
            // reduced coefficient the same way build_random_database_
            // polynomial_eval_form did internally: reduce_mod(raw, p).
            int64_t db_reduced_i = reduce_mod(db.raw_values[static_cast<size_t>(i)], params.plaintext_modulus);
            int64_t expected = (message.reduced * db_reduced_i) % params.plaintext_modulus;
            EXPECT_EQ(decrypted[i], expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i
                << ": message.reduced=" << message.reduced << " db_reduced_i=" << db_reduced_i
                << " expected=" << expected << " got=" << decrypted[i];

            // Check 2: overflow (true signed product vs. centered decode).
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
