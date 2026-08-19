// test_full_scoring_with_cluster_selection.cpp
//
// Combines test_rlwe_scoring_dot_product.cpp and test_rgsw_multiplication.cpp
// into the full Step 5-7 pipeline for one query:
//
//   1. Build l query values (+ their CRT components), each key-switched to
//      RLWE then converted to eval form ONCE PER COMPONENT RING. Reused
//      across every cluster below -- the query is the same regardless of
//      which cluster's database it's being scored against, so converting it
//      to eval form once and reusing it avoids num_clusters redundant
//      forward NTTs per query term (per ring).
//   2. Build num_clusters selector bits: all 0 except a 1 at
//      params.desired_cluster_index. Each gadget-encrypted as LWE, then
//      key-switched to RGSW. UNCHANGED by CRT -- exactly one selector set
//      regardless of num_component_rings, reused for every ring below.
//   3. For each cluster c:
//        a. Sample l database polynomials' raw values ONCE (shared across
//           rings -- CRT splits the SAME underlying data, not independently
//           sampled data per ring), then CRT-split each into its
//           per-component-ring eval form.
//        b. For each component ring: score_c[ring] = sum_j query[ring][j] *
//           db_c[ring][j] -- multiply AND sum each term in eval form, then
//           convert the per-cluster sum back to coefficient form once.
//        c. masked_c[ring] = RGSW_c * score_c[ring] -- zero unless
//           c == desired_cluster_index. SAME rgsw_ct for every ring.
//   4. final_result[ring] = sum_c masked_c[ring] (coefficient form), one
//      accumulator per component ring.
//   5. Decrypt each ring's final_result, recompose (a no-op when r==1), and
//      verify the recomposed result equals the dot product of the query
//      against ONLY the desired cluster's database -- every other cluster's
//      contribution should have been masked to exactly zero.
//
// Parameterized over TWO Params factories, run via the SAME test body --
// same reasoning and same two-checks-after-recomposition structure as
// test_rlwe_scoring_dot_product.cpp.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_cluster_selection

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

class FullScoringWithClusterSelection : public ::testing::TestWithParam<Params (*)()> {};

/// score_c = sum_j query[j] * db[j], for ONE component ring. Called once per
/// ring per cluster below -- UNCHANGED from the non-CRT version, since the
/// CRT dimension is handled entirely by which query_eval/db_cluster it's
/// called with, not by anything inside this function.
RLWECT compute_cluster_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                              const std::vector<DatabasePolynomialEvalForm>& db_cluster) {
    RLWECTEvalForm score_eval(ctx.rlwe_param); // zero-initialized accumulator, eval form
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_cluster[j].poly_eval);
        score_eval.add(score_eval, product_eval); // accumulate in eval form
    }
    return RLWECT(score_eval); // single eval -> coef conversion, once per cluster per ring
}

} // namespace

TEST_P(FullScoringWithClusterSelection, OnlyDesiredClusterContributes) {
    Params params = GetParam()();

    // This test loops over clusters but not splits -- pin database_size = n
    // locally so splits_per_cluster stays 1, independent of whatever "real"
    // (multi-split) database_size the factory uses. See
    // test_full_scoring_with_splits.cpp for the version that also loops
    // over splits.
    params.database_size = params.n * params.num_clusters;
    params.derive_dependent_parameters();

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully; "
        << "see Params::make_test_params()/make_test_params_component_rings().";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single cluster's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 5; // each iteration does num_clusters * embedding_length * r
                                       // multiplications plus num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Step 4/5 (embedding side): l query messages + their CRT
        // components, key-switched to RLWE once PER COMPONENT RING, reused
        // for every cluster below. --------------------------------------------------
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

        std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            query_eval[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.embedding_length));
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(
                    message_components_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)],
                    ctx.component_encodings[static_cast<size_t>(ring)]);
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct); // coef -> eval
            }
        }

        // --- Step 4/5 (cluster-selection side): UNCHANGED by CRT -- exactly
        // one selector set, reused for every ring below. --------------------------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Step 6/7: per-cluster, per-ring score, mask, accumulate. ---------------
        std::vector<std::vector<int64_t>> desired_cluster_raw_values; // [j][i] -- the TRUE raw
                                                                        // values, shared by both
                                                                        // rings (CRT splits the
                                                                        // SAME data, not different
                                                                        // data per ring)

        std::vector<RLWECT> final_result; // [ring], zero-initialized accumulators
        final_result.reserve(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            final_result.emplace_back(ctx.rlwe_param);
        }

        for (int64_t c = 0; c < params.num_clusters; ++c) {
            // Sample this cluster's raw database values ONCE, then CRT-split
            // each into its per-ring eval form -- crt_split_database_polynomial_eval_form
            // guarantees both rings come from the SAME raw_values.
            std::vector<std::vector<int64_t>> cluster_raw_values(static_cast<size_t>(params.embedding_length));
            std::vector<std::vector<DatabasePolynomialEvalForm>> db_eval_per_j(
                static_cast<size_t>(params.embedding_length)); // [j][ring]

            for (int64_t j = 0; j < params.embedding_length; ++j) {
                std::vector<int64_t> raw(static_cast<size_t>(params.n));
                for (int64_t i = 0; i < params.n; ++i) {
                    raw[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
                }
                db_eval_per_j[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
                cluster_raw_values[static_cast<size_t>(j)] = std::move(raw);
            }

            if (c == params.desired_cluster_index) {
                desired_cluster_raw_values = cluster_raw_values;
            }

            for (int64_t ring = 0; ring < r; ++ring) {
                std::vector<DatabasePolynomialEvalForm> db_cluster_ring;
                db_cluster_ring.reserve(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    db_cluster_ring.push_back(std::move(db_eval_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)]));
                }

                RLWECT score_c_ring = compute_cluster_score(ctx, query_eval[static_cast<size_t>(ring)], db_cluster_ring);

                RLWECT masked_c_ring(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)].mul(masked_c_ring, score_c_ring); // SAME rgsw_ct for every ring

                final_result[static_cast<size_t>(ring)].add(final_result[static_cast<size_t>(ring)], masked_c_ring);
            }
        }

        // --- Step 8: decrypt each ring, recompose (a no-op when r==1),
        // verify. --------------------------------------------------------------------------
        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            decrypted_per_ring.push_back(secret.rlwe_sk->decrypt_vector(
                final_result[static_cast<size_t>(ring)], ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            int64_t true_sum = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                true_sum += messages[static_cast<size_t>(j)].raw *
                            desired_cluster_raw_values[static_cast<size_t>(j)][static_cast<size_t>(i)];
            }

            // Check 1: mod-(combined) consistency.
            int64_t expected = reduce_mod(true_sum, combined_modulus);
            EXPECT_EQ(recomposed, expected)
                << "Mod-p mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): expected=" << expected << " got=" << recomposed;

            // Check 2: overflow.
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            EXPECT_EQ(decoded_signed, true_sum)
                << "Overflow/corruption on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                << " (recomposed residue was " << recomposed << ").";
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, FullScoringWithClusterSelection,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });