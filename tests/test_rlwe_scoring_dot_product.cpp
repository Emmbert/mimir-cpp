// test_rlwe_scoring_dot_product.cpp
//
// Tests the actual protocol operation one split's score computation reduces
// to (Step 6 of the protocol): given embedding_length (l) query values and l
// database polynomials, compute
//
//   score = sum_{j=0}^{l-1} RLWE(query[j]) * db_poly[j]
//
// Each term is multiplied AND summed entirely in eval/NTT form: query
// key-switched to RLWE then converted to eval form; database polynomial
// built directly in eval form via build_random_database_polynomial_eval_form;
// products accumulated via repeated RLWECTEvalForm::add; a single
// eval->coef conversion at the very end, right before decryption.
//
// History: an earlier version of this test summed in coefficient form
// instead, converting every term back individually before adding. That
// workaround existed because PolynomialEvalFormLongInteger::add (in
// FHE-Deck's polynomial.cpp) was missing its modular reduction --
// `out[i] = a[i] + b[i]` instead of `out[i] = (a[i] + b[i]) % modulus` --
// which corrupted results deterministically once more than 2 terms were
// summed in eval form. That bug has since been fixed (locally, in the
// fhe-deck-core checkout this project builds against), so this test now
// uses the straightforward, fully-eval-form accumulation.
//
// Checks two things per coefficient, same distinction as the single-
// multiplication tests:
//   1. Mod-p consistency: decrypted[i] == (sum_j query[j].reduced *
//      db[j].raw_values[i] reduced) mod plaintext_modulus.
//   2. No overflow: the TRUE (un-reduced) sum of products, decoded via the
//      centered convention, must equal decrypted[i] exactly -- this is what
//      dot_product_can_overflow's precondition check guards against ever
//      failing.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_scoring_dot_product

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

TEST(RlweScoringDotProduct, SumOfLProductsMatchesTrueDotProduct) {
    Params params = Params::make_test_params();

    // This test scores a single split only -- it never loops over clusters
    // or splits -- so pin database_size = n here (splits_per_cluster == 1),
    // independent of whatever "real" (multi-split) database_size
    // Params::make_test_params() uses for the rest of the protocol.
    params.database_size = params.n;
    params.derive_dependent_parameters();

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible: the worst-case dot-product magnitude is embedding_length * "
        << "max_abs_embedding_value(params)^2 = " << params.embedding_length << " * "
        << max_abs_embedding_value(params) << "^2, which is >= plaintext_modulus/2 ("
        << (params.plaintext_modulus / 2) << "). Increase plaintext_modulus or decrease "
        << "embedding_precision/embedding_length so that embedding_length * "
        << "max_abs_embedding_value(params)^2 < plaintext_modulus/2.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Build l query values, each key-switched to RLWE and converted
        // to eval form. ---------------------------------------------------
        std::vector<SignedValue> messages;
        std::vector<RLWECTEvalForm> query_eval;
        messages.reserve(static_cast<size_t>(params.embedding_length));
        query_eval.reserve(static_cast<size_t>(params.embedding_length));

        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);

            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(m.reduced, ctx.encoding);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
            query_eval.emplace_back(rlwe_ct); // coef-form -> eval-form
        }

        // --- Build l database polynomials, directly in eval form. --------
        std::vector<DatabasePolynomialEvalForm> db;
        db.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            db.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
        }

        // --- score = sum_j query_eval[j] * db[j], entirely in eval form. --
        RLWECTEvalForm score_eval(ctx.rlwe_param);
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            RLWECTEvalForm product(ctx.rlwe_param);
            query_eval[static_cast<size_t>(j)].mul(product, *db[static_cast<size_t>(j)].poly_eval);
            score_eval.add(score_eval, product);
        }

        RLWECT score_ct(score_eval); // eval-form -> coef-form, once, for decryption
        Vector decrypted = secret.rlwe_sk->decrypt_vector(score_ct, ctx.encoding);
        std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            // Check 1: mod-p consistency.
            int64_t expected = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                int64_t db_reduced_i_j =
                    reduce_mod(db[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)], params.plaintext_modulus);
                expected = (expected + messages[static_cast<size_t>(j)].reduced * db_reduced_i_j) % params.plaintext_modulus;
            }
            EXPECT_EQ(decrypted[i], expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i
                << ": expected=" << expected << " got=" << decrypted[i];

            // Check 2: overflow. TRUE (un-reduced) sum of products vs.
            // centered decode of the actual decrypted result.
            int64_t true_sum = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                true_sum += messages[static_cast<size_t>(j)].raw * db[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            }
            int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
            EXPECT_EQ(decoded_signed, true_sum)
                << "Overflow on iteration " << iter << ", coefficient " << i
                << ": true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                << " (mod-p residue was " << decrypted[i] << "). This means "
                << "embedding_precision/embedding_length/plaintext_modulus allow dot products "
                << "whose magnitude exceeds plaintext_modulus/2.";
        }
    }
}