// test_rlwe_scoring_dot_product_coef_sum.cpp
//
// DIAGNOSTIC variant of test_rlwe_scoring_dot_product.cpp. Identical in every
// way EXCEPT one: instead of accumulating the l per-term products via
// repeated self-aliased RLWECTEvalForm::add(...) calls in eval form, each
// product is converted back to coefficient form immediately and summed via
// RLWECT::add(...) instead.
//
// This isolates exactly one variable: does the bug (deterministic failure
// starting at embedding_length >= 3, every coefficient, every iteration)
// depend on WHERE the summation happens?
//
//   - If THIS test passes reliably at embedding_length >= 3 while the
//     eval-form-accumulation version fails: the bug is specifically in
//     repeated self-aliased RLWECTEvalForm::add, not in the key switch or in
//     RLWECTEvalForm::mul (both of those are exercised identically here).
//   - If this ALSO fails the same way: the bug is elsewhere (multiply, key
//     switch, or something shared), and this diagnostic doesn't localize it
//     further, but is useful to rule out.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_scoring_dot_product_coef_sum

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

} // namespace

TEST(RlweScoringDotProductCoefSum, SumOfLProductsMatchesTrueDotProduct) {
    Params params = Params::make_test_params();

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        std::vector<SignedValue> messages;
        std::vector<RLWECT> query_coef; // kept alive for the whole iteration
        messages.reserve(static_cast<size_t>(params.embedding_length));
        query_coef.reserve(static_cast<size_t>(params.embedding_length));

        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);

            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(m.reduced, ctx.encoding);
            query_coef.push_back(switch_to_rlwe(ctx, pub, lwe_ct));
        }

        std::vector<DatabasePolynomialEvalForm> db;
        db.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            db.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
        }

        // --- Multiply in eval form (per-term, already independently
        // verified correct), but sum in COEFFICIENT form. -------------------
        RLWECT score_ct(ctx.rlwe_param); // zero-initialized accumulator
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            RLWECTEvalForm query_j_eval(query_coef[static_cast<size_t>(j)]); // coef -> eval
            RLWECTEvalForm product_eval(ctx.rlwe_param);
            query_j_eval.mul(product_eval, *db[static_cast<size_t>(j)].poly_eval);

            RLWECT product_ct(product_eval); // eval -> coef, immediately
            score_ct.add(score_ct, product_ct); // accumulate in coefficient form
        }

        Vector decrypted = secret.rlwe_sk->decrypt_vector(score_ct, ctx.encoding);
        std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                int64_t db_reduced_i_j =
                    reduce_mod(db[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)], params.plaintext_modulus);
                expected = (expected + messages[static_cast<size_t>(j)].reduced * db_reduced_i_j) % params.plaintext_modulus;
            }
            EXPECT_EQ(decrypted[i], expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i
                << ": expected=" << expected << " got=" << decrypted[i];

            int64_t true_sum = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                true_sum += messages[static_cast<size_t>(j)].raw * db[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            }
            int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
            EXPECT_EQ(decoded_signed, true_sum)
                << "Overflow/corruption on iteration " << iter << ", coefficient " << i
                << ": true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                << " (mod-p residue was " << decrypted[i] << ").";
        }
    }
}
