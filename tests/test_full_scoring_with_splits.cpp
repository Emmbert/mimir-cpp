// test_full_scoring_with_splits.cpp
//
// Extends test_full_scoring_with_cluster_selection.cpp with the splits
// dimension: if a cluster holds more documents than fit in one ring's worth
// of n coefficients, it's divided into splits_per_cluster splits, and every
// calculation (per-cluster scoring, RGSW masking) happens independently per
// split, per component ring. The client ends up with splits_per_cluster
// final RLWE ciphertexts PER component ring instead of one -- each
// (split, ring) pair corresponds to one chunk of documents within whichever
// cluster was selected, decrypted separately and recomposed (a no-op when
// r==1) before checking.
//
// Key structural facts this test exercises:
//   - splits_per_cluster = ceil(cluster_size / n). The factory's
//     database_size makes cluster_size > n and splits_per_cluster > 1 here;
//     every other test in this suite locally pins database_size = n to stay
//     in the simpler single-split case.
//   - The query (query_eval, one set PER component ring) and the RGSW
//     cluster-selection ciphertexts (rgsw_ct, UNCHANGED by CRT -- exactly
//     one set regardless of num_component_rings) do NOT depend on split --
//     both built ONCE per iteration, reused across every split AND (for
//     rgsw_ct) every ring.
//   - Only the database polynomials are split-specific, AND (for CRT)
//     component-ring-specific -- but both rings of a given (cluster, split)
//     come from the SAME sampled raw values, via
//     crt_split_database_polynomial_eval_form -- see
//     test_full_scoring_with_cluster_selection.cpp for why this matters.
//   - final_result becomes final_result[ring][split] -- splits_per_cluster
//     RLWE ciphertexts per component ring, each accumulated by summing
//     masked_{c,s,ring} over all clusters c, for that fixed (s, ring).
//
// Parameterized over TWO Params factories, run via the SAME test body.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_splits

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

class FullScoringWithSplits : public ::testing::TestWithParam<Params (*)()> {};

/// score = sum_j query[j] * db[j], for ONE component ring, ONE
/// (cluster, split) pair. UNCHANGED from the non-CRT version -- the CRT
/// dimension is handled entirely by which query_eval/db_split it's called
/// with.
RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                            const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

} // namespace

TEST_P(FullScoringWithSplits, EachSplitProducesCorrectResultForDesiredCluster) {
    Params params = GetParam()();
    // Uses database_size (and therefore cluster_size / splits_per_cluster)
    // exactly as the factory sets it -- this is the one test in the suite
    // that exercises the "real" (multi-split) database size; every other
    // test locally pins database_size = n to stay in the simpler
    // single-split case.

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully.";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);
    /*if (params.splits_per_cluster == 1) {
        GTEST_SKIP() << "This test needs splits_per_cluster > 1 to exercise multiple splits; got "
                     << "cluster_size=" << params.cluster_size << ", n=" << params.n
                     << ", splits_per_cluster=" << params.splits_per_cluster
                     << ". Increase database_size or decrease num_clusters/n.";
    }*/

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single split's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 3; // each iteration does
                                       // num_clusters * splits_per_cluster * embedding_length * r
                                       // multiplications plus num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Query: built once PER COMPONENT RING, reused across every
        // cluster AND every split. --------------------------------------------------
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
                query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct);
            }
        }

        // --- Cluster-selection RGSW ciphertexts: UNCHANGED by CRT -- built
        // once per cluster, reused across every split AND every ring. ------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Per (cluster, split, ring): score, mask, accumulate into
        // final_result[ring][s]. Desired cluster's raw database values kept
        // per split (shared across rings -- CRT splits the SAME data). -------------
        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values( // [s][j] -> vector<int64_t> length n
            static_cast<size_t>(params.splits_per_cluster));

        std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r)); // [ring][s]
        for (int64_t ring = 0; ring < r; ++ring) {
            final_result[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result[static_cast<size_t>(ring)].emplace_back(ctx.rlwe_param);
            }
        }

        for (int64_t c = 0; c < params.num_clusters; ++c) {
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                std::vector<std::vector<int64_t>> split_raw_values(static_cast<size_t>(params.embedding_length));
                std::vector<std::vector<DatabasePolynomialEvalForm>> db_eval_per_j( // [j][ring]
                    static_cast<size_t>(params.embedding_length));

                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    std::vector<int64_t> raw(static_cast<size_t>(params.n));
                    for (int64_t i = 0; i < params.n; ++i) {
                        raw[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
                    }
                    db_eval_per_j[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
                    split_raw_values[static_cast<size_t>(j)] = std::move(raw);
                }

                if (c == params.desired_cluster_index) {
                    desired_cluster_raw_values[static_cast<size_t>(s)] = split_raw_values;
                }

                for (int64_t ring = 0; ring < r; ++ring) {
                    std::vector<DatabasePolynomialEvalForm> db_split_ring;
                    db_split_ring.reserve(static_cast<size_t>(params.embedding_length));
                    for (int64_t j = 0; j < params.embedding_length; ++j) {
                        db_split_ring.push_back(
                            std::move(db_eval_per_j[static_cast<size_t>(j)][static_cast<size_t>(ring)]));
                    }

                    RLWECT score = compute_split_score(ctx, query_eval[static_cast<size_t>(ring)], db_split_ring);

                    RLWECT masked(ctx.rlwe_param);
                    rgsw_ct[static_cast<size_t>(c)].mul(masked, score); // SAME rgsw_ct for every ring

                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)].add(
                        final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)], masked);
                }
            }
        }

        // --- Decrypt and verify EACH split's result independently, across
        // both rings, then recompose (a no-op when r==1). --------------------------
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            std::vector<Vector> decrypted_per_ring;
            decrypted_per_ring.reserve(static_cast<size_t>(r));
            for (int64_t ring = 0; ring < r; ++ring) {
                decrypted_per_ring.push_back(secret.rlwe_sk->decrypt_vector(
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                    ctx.component_encodings[static_cast<size_t>(ring)]));
            }

            const auto& raw_values_for_split = desired_cluster_raw_values[static_cast<size_t>(s)];

            for (int64_t i = 0; i < params.n; ++i) {
                int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                               : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                                params.comp_ring_modulus);

                int64_t true_sum = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    true_sum += messages[static_cast<size_t>(j)].raw *
                                raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)];
                }

                int64_t expected = reduce_mod(true_sum, combined_modulus);
                EXPECT_EQ(recomposed, expected)
                    << "Mod-p mismatch on iteration " << iter << ", split " << s << ", coefficient " << i
                    << " (r=" << r << "): expected=" << expected << " got=" << recomposed;

                int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
                EXPECT_EQ(decoded_signed, true_sum)
                    << "Overflow/corruption on iteration " << iter << ", split " << s << ", coefficient " << i
                    << " (r=" << r << "): true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                    << " (recomposed residue was " << recomposed << ").";
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, FullScoringWithSplits,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });