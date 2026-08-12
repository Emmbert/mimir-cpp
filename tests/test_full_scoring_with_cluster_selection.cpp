// test_full_scoring_with_cluster_selection.cpp
//
// Combines test_rlwe_scoring_dot_product.cpp and test_rgsw_multiplication.cpp
// into the full Step 5-7 pipeline for one query:
//
//   1. Build l query values, each key-switched to RLWE then converted to
//      eval form ONCE. Reused across every cluster below -- the query is the
//      same regardless of which cluster's database it's being scored
//      against, so converting it to eval form once and reusing it avoids
//      num_clusters redundant forward NTTs per query term.
//   2. Build num_clusters selector bits: all 0 except a 1 at
//      params.desired_cluster_index. Each gadget-encrypted as LWE, then
//      key-switched to RGSW.
//   3. For each cluster c:
//        a. Build l database polynomials for that cluster, in eval form.
//        b. score_c = sum_j query[j] * db_c[j] -- multiply AND sum each term
//           in eval form (safe: the missing-modular-reduction bug in
//           PolynomialEvalFormLongInteger::add has been fixed), then convert
//           the final per-cluster sum back to coefficient form once.
//        c. masked_c = RGSW_c * score_c -- zero unless c == desired_cluster_index.
//   4. final_result = sum_c masked_c (coefficient form).
//   5. Decrypt final_result and verify it equals the dot product of the
//      query against ONLY the desired cluster's database -- every other
//      cluster's contribution should have been masked to exactly zero.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_cluster_selection

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

/// score_c = sum_j query[j] * db[j]. Multiplies each term AND sums in eval
/// form -- safe now that the missing-modular-reduction bug in
/// PolynomialEvalFormLongInteger::add has been fixed locally (see
/// test_rlwe_scoring_dot_product.cpp's history: that bug used to corrupt
/// results once more than 2 terms were summed in eval form, forcing a
/// workaround of converting every term back to coefficient form before
/// summing). With the fix in place, only ONE eval->coef conversion happens
/// per cluster (at the very end), instead of one per term.
RLWECT compute_cluster_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                              const std::vector<DatabasePolynomialEvalForm>& db_cluster) {
    RLWECTEvalForm score_eval(ctx.rlwe_param); // zero-initialized accumulator, eval form
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_cluster[j].poly_eval);
        score_eval.add(score_eval, product_eval); // accumulate in eval form
    }
    return RLWECT(score_eval); // single eval -> coef conversion, once per cluster
}

} // namespace

TEST(FullScoringWithClusterSelection, OnlyDesiredClusterContributes) {
    Params params = Params::make_test_params();

    // This test loops over clusters but not splits -- pin database_size = n
    // locally so splits_per_cluster stays 1, independent of whatever "real"
    // (multi-split) database_size Params::make_test_params() uses. See
    // test_full_scoring_with_splits.cpp for the version that also loops over
    // splits and uses the real database_size directly.
    params.database_size = params.n*params.num_clusters;
    params.derive_dependent_parameters();

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully; "
        << "see Params::make_test_params().";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single cluster's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 5; // each iteration does num_clusters * embedding_length
                                       // multiplications plus num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Step 4/5 (embedding side): l query values, key-switched to
        // RLWE once, reused for every cluster below. ------------------------
        std::vector<SignedValue> messages;
        std::vector<RLWECTEvalForm> query_eval;
        messages.reserve(static_cast<size_t>(params.embedding_length));
        query_eval.reserve(static_cast<size_t>(params.embedding_length));

        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);

            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(m.reduced, ctx.encoding);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
            query_eval.emplace_back(rlwe_ct); // coef -> eval, once, reused for every cluster below
        }

        // --- Step 4/5 (cluster-selection side): num_clusters selector bits,
        // all 0 except a 1 at desired_cluster_index, each gadget-encrypted
        // and key-switched to RGSW. ------------------------------------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Step 6/7: per-cluster score, mask, accumulate. -----------------
        // The desired cluster's raw database values are kept so we can
        // compute the expected result afterwards.
        std::vector<std::vector<int64_t>> desired_cluster_raw_values; // [j][i]

        RLWECT final_result(ctx.rlwe_param); // zero-initialized accumulator
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            std::vector<DatabasePolynomialEvalForm> db_cluster;
            db_cluster.reserve(static_cast<size_t>(params.embedding_length));
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                db_cluster.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
            }

            if (c == params.desired_cluster_index) {
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    desired_cluster_raw_values.push_back(db_cluster[static_cast<size_t>(j)].raw_values);
                }
            }

            RLWECT score_c = compute_cluster_score(ctx, query_eval, db_cluster);

            RLWECT masked_c(ctx.rlwe_param);
            rgsw_ct[static_cast<size_t>(c)].mul(masked_c, score_c);

            final_result.add(final_result, masked_c);
        }

        // --- Step 8: decrypt and verify. -------------------------------------
        Vector decrypted = secret.rlwe_sk->decrypt_vector(final_result, ctx.encoding);
        std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t expected = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                int64_t db_reduced_i_j = reduce_mod(
                    desired_cluster_raw_values[static_cast<size_t>(j)][static_cast<size_t>(i)],
                    params.plaintext_modulus);
                expected = (expected + messages[static_cast<size_t>(j)].reduced * db_reduced_i_j) % params.plaintext_modulus;
            }
            EXPECT_EQ(decrypted[i], expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i
                << ": expected=" << expected << " got=" << decrypted[i];

            int64_t true_sum = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                true_sum += messages[static_cast<size_t>(j)].raw *
                            desired_cluster_raw_values[static_cast<size_t>(j)][static_cast<size_t>(i)];
            }
            int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
            EXPECT_EQ(decoded_signed, true_sum)
                << "Overflow/corruption on iteration " << iter << ", coefficient " << i
                << ": true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                << " (mod-p residue was " << decrypted[i] << ").";
        }
    }
}