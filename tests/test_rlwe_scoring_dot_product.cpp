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
// built directly in eval form; products accumulated via repeated
// RLWECTEvalForm::add; a single eval->coef conversion at the very end,
// right before decryption.
//
// Parameterized over TWO Params factories, run via the SAME test body --
// same reasoning as the earlier CRT tests. The key structural point for
// CRT: the SUM ITSELF happens separately per component ring -- l terms
// summed in eval form for ring 0, l terms summed in eval form for ring 1 --
// each ring's full sum is only decrypted (never recomposed) until AFTER
// both sums are complete. Recomposing term-by-term instead of after the
// full sum would be wrong: CRT recomposition is only meaningful on a
// genuine residue pair, not on partial sums that haven't reached their
// final value yet.
//
// Checks two things per coefficient, after recomposition (a no-op when
// r==1), same distinction as the single-multiplication tests -- see
// test_rlwe_plaintext_multiplication.cpp's header for why both checks only
// make sense post-recomposition, against the combined modulus, not
// per-component:
//   1. Mod-(combined) consistency: recomposed[i] == (true, unreduced sum of
//      raw products) reduced mod the combined modulus.
//   2. No overflow: the TRUE (un-reduced) sum of products, decoded via the
//      centered convention (against the combined modulus), must equal
//      recomposed[i] exactly -- this is what dot_product_can_overflow's
//      precondition check guards against ever failing. That check itself
//      is UNCHANGED by CRT -- always evaluated against plaintext_modulus
//      (the required lower bound); "safe against plaintext_modulus" implies
//      "safe against the larger-or-equal combined modulus" automatically.
//
// History: an earlier version of this test summed in coefficient form
// instead, converting every term back individually before adding. That
// workaround existed because PolynomialEvalFormLongInteger::add (in
// FHE-Deck's polynomial.cpp) was missing its modular reduction -- fixed
// since, so this test uses the straightforward, fully-eval-form
// accumulation.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_rlwe_scoring_dot_product

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

class RlweScoringDotProduct : public ::testing::TestWithParam<Params (*)()> {};

} // namespace

TEST_P(RlweScoringDotProduct, SumOfLProductsMatchesTrueDotProduct) {
    Params params = GetParam()();

    // This test scores a single split only -- it never loops over clusters
    // or splits -- so pin database_size = n here (splits_per_cluster == 1),
    // independent of whatever "real" (multi-split) database_size the
    // factory uses for the rest of the protocol. Re-validates CRT fields
    // too (unchanged by this override).
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
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 20;
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Build l query messages and their CRT components (message_components_per_j[j][ring]). ---
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<std::vector<int64_t>> message_components_per_j(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            messages[static_cast<size_t>(j)] = sample_signed_value(params, rng);
            if (r == 1) {
                message_components_per_j[static_cast<size_t>(j)] = {messages[static_cast<size_t>(j)].reduced};
            } else {
                int64_t canonical = reduce_mod(messages[static_cast<size_t>(j)].raw, combined_modulus);
                auto [c1, c2] = crt_split(canonical, params.comp_ring_modulus);
                message_components_per_j[static_cast<size_t>(j)] = {c1, c2};
            }
        }

        // --- Build l database polynomials (raw values), and their CRT
        // eval-form splits (db_eval_per_j[j][ring]). -----------------------------------
        std::vector<std::vector<int64_t>> db_raw_values_per_j(static_cast<size_t>(params.embedding_length));
        std::vector<std::vector<DatabasePolynomialEvalForm>> db_eval_per_j(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            std::vector<int64_t> raw(static_cast<size_t>(params.n));
            for (int64_t i = 0; i < params.n; ++i) {
                raw[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
            }
            db_eval_per_j[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
            db_raw_values_per_j[static_cast<size_t>(j)] = std::move(raw);
        }

        // --- Per component ring: score = sum_j query_eval[j] * db_eval[j][ring],
        // entirely in eval form. Each ring's sum is complete and decrypted
        // BEFORE any recomposition happens -- see file header for why
        // recomposing term-by-term would be wrong. ------------------------------------
        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));

        for (int64_t ring = 0; ring < r; ++ring) {
            RLWECTEvalForm score_eval(ctx.rlwe_param);

            for (int64_t j = 0; j < params.embedding_length; ++j) {
                LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(
                    message_components_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)],
                    ctx.component_encodings[static_cast<size_t>(ring)]);
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                RLWECTEvalForm query_eval_j(rlwe_ct); // coef-form -> eval-form

                RLWECTEvalForm product(ctx.rlwe_param);
                query_eval_j.mul(product, *db_eval_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)].poly_eval);
                score_eval.add(score_eval, product);
            }

            RLWECT score_ct(score_eval); // eval-form -> coef-form, once, for decryption
            decrypted_per_ring.push_back(
                secret.rlwe_sk->decrypt_vector(score_ct, ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        // --- Recompose (a no-op when r==1) and check every coefficient. -----------
        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            int64_t true_sum = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                true_sum += messages[static_cast<size_t>(j)].raw *
                            db_raw_values_per_j[static_cast<size_t>(j)][static_cast<size_t>(i)];
            }

            // Check 1: mod-(combined) consistency.
            int64_t expected = reduce_mod(true_sum, combined_modulus);
            EXPECT_EQ(recomposed, expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): expected=" << expected << " got=" << recomposed;

            // Check 2: overflow.
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            EXPECT_EQ(decoded_signed, true_sum)
                << "Overflow on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                << " (recomposed residue was " << recomposed << "). This means "
                << "embedding_precision/embedding_length/plaintext_modulus allow dot products "
                << "whose magnitude exceeds combined_modulus/2.";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, RlweScoringDotProduct,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });
