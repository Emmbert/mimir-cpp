// test_rlwe_plaintext_multiplication_ntt.cpp
//
// Same test as test_rlwe_plaintext_multiplication.cpp (same sampling, same
// mod-p consistency check, same overflow check, same CRT handling), but
// performs the actual ciphertext x plaintext-polynomial multiplication in
// NTT/eval form instead of coefficient form.
//
// The database polynomial is built directly in eval/NTT form via
// crt_split_database_polynomial_eval_form (db_polynomial.hpp) -- there is
// no coefficient-form Polynomial in this file at all for the database side;
// that intermediate lives and dies entirely inside that builder function.
// This matches how the real database will eventually be stored (Step 1 of
// the protocol: "All polynomials are transferred into NTT representation") --
// ServerDatabase::build will call the same underlying builder for every
// polynomial in every split of every cluster (and, for CRT, once per
// component ring), so the database is in eval form from the moment it's
// created, never converted at query time.
//
// The ciphertext side still needs an explicit conversion per component
// ring, following the reference perf_test_sequential() sample:
//
//   RLWECTEvalForm ct_eval(ct);              // ciphertext coef-form -> eval-form
//   RLWECTEvalForm product(rlwe_param);
//   ct_eval.mul(product, *database_entry_eval_form);   // multiply in eval form
//   RLWECT out(product);                     // eval-form -> coef-form, for decrypt
//
// Parameterized over TWO Params factories, run via the SAME test body --
// see test_rlwe_plaintext_multiplication.cpp's header for the full
// reasoning (both checks happen only AFTER recomposition, against the
// combined modulus, not per-component).
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_plaintext_multiplication_ntt

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

class RlwePlaintextMultiplicationNtt : public ::testing::TestWithParam<Params (*)()> {};

LWECT encrypt_lwe(const ClientSecretMaterial& secret, int64_t message, const PlaintextEncoding& encoding) {
    return secret.lwe_sk->encode_and_encrypt(message, encoding);
}

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

} // namespace

TEST_P(RlwePlaintextMultiplicationNtt, ScalarTimesRandomDatabasePolynomialEvalForm) {
    Params params = GetParam()();

    // Same precondition as the coefficient-form version -- overflow is a
    // property of the parameters/values, not of which multiplication path
    // is used, so this check doesn't change, and is UNCHANGED by CRT (see
    // test_rlwe_plaintext_multiplication.cpp's header for why).
    ASSERT_FALSE(products_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << ") and plaintext_modulus ("
        << params.plaintext_modulus << ") are incompatible: the worst-case product magnitude is "
        << max_abs_embedding_value(params) << "^2 = "
        << (max_abs_embedding_value(params) * max_abs_embedding_value(params))
        << ", which is >= plaintext_modulus/2 (" << (params.plaintext_modulus / 2) << "). "
        << "Increase plaintext_modulus or decrease embedding_precision so that "
        << "max_abs_embedding_value(params)^2 < plaintext_modulus/2.";

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

        // Database: n independent raw values, same as the coefficient-form
        // test -- sampled once, then split per component ring below.
        std::vector<int64_t> db_raw_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            db_raw_values[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
        }

        std::vector<int64_t> message_components;
        if (r == 1) {
            message_components = {message.reduced};
        } else {
            int64_t canonical = reduce_mod(message.raw, combined_modulus);
            auto [c1, c2] = crt_split(canonical, params.comp_ring_modulus);
            message_components = {c1, c2};
        }

        // --- Database, directly in eval/NTT form, one entry per component
        // ring. -----------------------------------------------------------------------
        std::vector<DatabasePolynomialEvalForm> db_per_ring =
            crt_split_database_polynomial_eval_form(ctx, params, db_raw_values);

        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));

        for (int64_t ring = 0; ring < r; ++ring) {
            LWECT lwe_ct = encrypt_lwe(secret, message_components[static_cast<size_t>(ring)],
                                       ctx.component_encodings[static_cast<size_t>(ring)]);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);

            // Ciphertext: coefficient form -> eval form.
            RLWECTEvalForm rlwe_ct_eval(rlwe_ct);

            // Multiply directly in eval form, against THIS ring's database.
            RLWECTEvalForm product_eval(ctx.rlwe_param);
            rlwe_ct_eval.mul(product_eval, *db_per_ring[static_cast<size_t>(ring)].poly_eval);

            // Back to coefficient form for decryption.
            RLWECT product_ct(product_eval);

            decrypted_per_ring.push_back(
                secret.rlwe_sk->decrypt_vector(product_ct, ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        // --- Recompose (a no-op when r==1) and check every coefficient.
        // Identical structure to the coefficient-form test -- see its file
        // header for why both checks only happen after recomposition. -------------
        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            // Check 1: mod-(combined) consistency.
            int64_t expected = reduce_mod(message.raw * db_raw_values[static_cast<size_t>(i)], combined_modulus);
            EXPECT_EQ(recomposed, expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): message.raw=" << message.raw
                << " db_raw_values[i]=" << db_raw_values[static_cast<size_t>(i)] << " expected=" << expected
                << " got=" << recomposed;

            // Check 2: overflow, against the recomposed value and the
            // COMBINED modulus.
            int64_t true_product = message.raw * db_raw_values[static_cast<size_t>(i)];
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            EXPECT_EQ(decoded_signed, true_product)
                << "Overflow on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): raw message=" << message.raw << " raw db value=" << db_raw_values[static_cast<size_t>(i)]
                << " true product=" << true_product << " but decoded signed value=" << decoded_signed
                << " (recomposed residue was " << recomposed << "). This means embedding_precision/"
                << "plaintext_modulus allow products whose magnitude exceeds combined_modulus/2 -- increase "
                << "plaintext_modulus/comp_ring_modulus or decrease embedding_precision.";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, RlwePlaintextMultiplicationNtt,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });
