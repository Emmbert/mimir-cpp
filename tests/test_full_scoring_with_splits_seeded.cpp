// test_full_scoring_with_splits_seeded.cpp
//
// Same protocol, same checks as test_full_scoring_with_splits.cpp, but both
// the eval keys AND the query are built with seeds, wire-compressed, and
// reconstructed with NO secret key involved in reconstruction -- see
// seeded_eval_keys.hpp and seeded_query.hpp. Everything downstream of that
// (per-(cluster,split,ring) scoring, RGSW masking, cross-cluster
// accumulation, per-split decryption/recomposition/verification) is
// UNCHANGED from the non-seeded version -- this test exists specifically to
// confirm that swapping in reconstructed material doesn't change the
// protocol's correctness at all, which is exactly the property a real
// deployment depends on.
//
// Notably SIMPLER than the non-seeded version for the CRT dimension:
// build_seeded_query/reconstruct_query are already CRT-aware (they always
// were, by construction -- see seeded_query.hpp), so
// query.embedding_cts comes back as [ring][j] directly, with no manual
// per-ring encryption loop needed here at all. This test is really
// validating that THAT existing CRT-awareness produces correct results end
// to end, not re-implementing it.
//
// Key structural facts this test exercises (same as the non-seeded version):
//   - splits_per_cluster = ceil(cluster_size / n). The factory's
//     database_size makes splits_per_cluster > 1 here.
//   - The query (query_eval[ring]) and the RGSW cluster-selection
//     ciphertexts (rgsw_ct, UNCHANGED by CRT) do NOT depend on split --
//     built once per iteration, reused across every split.
//   - Only the database polynomials are split- and ring-specific, but both
//     rings of a given (cluster, split) come from the SAME sampled raw
//     values -- see test_full_scoring_with_cluster_selection.cpp.
//
// Parameterized over TWO Params factories, run via the SAME test body.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_splits_seeded

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class FullScoringWithSplitsSeeded : public ::testing::TestWithParam<Params (*)()> {};

/// score = sum_j query[j] * db[j], for ONE component ring, ONE
/// (cluster, split) pair. Identical to the non-seeded version -- nothing
/// about seeding OR CRT changes the scoring math itself.
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

TEST_P(FullScoringWithSplitsSeeded, EachSplitProducesCorrectResultUsingSeededEvalKeysAndQuery) {
    Params params = GetParam()();
    // Uses database_size exactly as the factory sets it -- same reasoning
    // as the non-seeded version: this is meant to exercise the "real"
    // (multi-split) database size.

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

    constexpr int kNumIterations = 3; // each iteration does a full eval-key seed+reconstruct,
                                       // a full query seed+reconstruct, plus
                                       // num_clusters * splits_per_cluster * embedding_length * r
                                       // multiplications and num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- Eval keys: seeded, wire-compressed, reconstructed. UNCHANGED
        // by CRT -- no dependency on plaintext modulus at all. -----------------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- Query: sample embedding values, seed-compress (build_seeded_query
        // handles CRT-splitting internally), reconstruct. -------------------------------
        std::vector<SignedValue> messages(static_cast<size_t>(params.embedding_length));
        std::vector<int64_t> embedding_values(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            messages[static_cast<size_t>(j)] = sample_signed_value(params, rng);
            // NOT messages[j].reduced (reduced mod plaintext_modulus) --
            // build_seeded_query CRT-splits whatever it receives directly,
            // and its contract requires values already canonical mod the
            // COMBINED modulus. Using plaintext_modulus here silently
            // differs from combined_modulus whenever the raw value is
            // negative (reduce_mod(-1, 71) = 70 != reduce_mod(-1, 110) =
            // 109), corrupting the CRT split relative to what verification
            // below expects.
            embedding_values[static_cast<size_t>(j)] = reduce_mod(messages[static_cast<size_t>(j)].raw, combined_modulus);
        }

        SeededQuery query_wire =
            build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);
        ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);
        ASSERT_EQ(static_cast<int64_t>(query.embedding_cts.size()), r);

        // --- Switch reconstructed embedding ciphertexts to RLWE/eval form,
        // PER COMPONENT RING, using the RECONSTRUCTED pub. Built once, reused
        // across every cluster AND every split -- same as the non-seeded version,
        // but query.embedding_cts is ALREADY [ring][j] here, no manual
        // per-ring encryption needed. -------------------------------------------------
        std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r)); // [ring][j]
        for (int64_t ring = 0; ring < r; ++ring) {
            const auto& embedding_cts_ring = query.embedding_cts[static_cast<size_t>(ring)];
            query_eval[static_cast<size_t>(ring)].reserve(embedding_cts_ring.size());
            for (const auto& lwe_ct : embedding_cts_ring) {
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct);
            }
        }

        // --- Switch reconstructed selector ciphertexts to RGSW, using the
        // RECONSTRUCTED pub. UNCHANGED by CRT -- built once per cluster,
        // reused across every split AND every ring. -----------------------------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(query.selector_cts.size());
        for (const auto& gadget_ct : query.selector_cts) {
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Everything below matches test_full_scoring_with_splits.cpp's
        // structure exactly: per-(cluster,split,ring) score, mask,
        // accumulate; per-split decrypt, recompose, verify. -------------------------
        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values( // [s][j]
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

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, FullScoringWithSplitsSeeded,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });