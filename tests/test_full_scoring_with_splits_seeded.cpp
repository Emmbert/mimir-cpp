// test_full_scoring_with_splits_seeded.cpp
//
// Same protocol, same checks as test_full_scoring_with_splits.cpp, but both
// the eval keys AND the query are built with seeds, wire-compressed, and
// reconstructed with NO secret key involved in reconstruction -- see
// seeded_eval_keys.hpp and seeded_query.hpp. Everything downstream of that
// (per-(cluster,split) scoring, RGSW masking, cross-cluster accumulation,
// per-split decryption and verification) is UNCHANGED from the non-seeded
// version -- this test exists specifically to confirm that swapping in
// reconstructed material doesn't change the protocol's correctness at all,
// which is exactly the property a real deployment depends on.
//
// Key structural facts this test exercises (same as the non-seeded version):
//   - splits_per_cluster = ceil(cluster_size / n), where
//     cluster_size = database_size / num_clusters. Params::make_test_params()
//     sets the "real" database_size so splits_per_cluster > 1 here.
//   - The query (query_eval) and the RGSW cluster-selection ciphertexts
//     (rgsw_ct) do NOT depend on split -- built once per iteration, reused
//     across every split.
//   - Only the database polynomials are split-specific.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_splits_seeded

#include <gtest/gtest.h>

#include <random>
#include <vector>

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

/// score = sum_j query[j] * db[j], multiplying AND summing entirely in eval
/// form, with a single eval->coef conversion at the end. Used once per
/// (cluster, split) pair. Identical to the non-seeded version -- nothing
/// about seeding changes the scoring math itself.
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

TEST(FullScoringWithSplitsSeeded, EachSplitProducesCorrectResultUsingSeededEvalKeysAndQuery) {
    Params params = Params::make_test_params();
    // Uses database_size exactly as Params::make_test_params() sets it --
    // same reasoning as the non-seeded version: this is the one test in the
    // suite meant to exercise the "real" (multi-split) database size.

    ASSERT_GT(params.num_clusters, 1)
        << "This test needs num_clusters > 1 to exercise cluster selection meaningfully.";
    ASSERT_GE(params.desired_cluster_index, 0);
    ASSERT_LT(params.desired_cluster_index, params.num_clusters);
    if (params.splits_per_cluster==1) {
        GTEST_SKIP() << "This test needs splits_per_cluster > 1 to exercise multiple splits; got "
        << "cluster_size=" << params.cluster_size << ", n=" << params.n
        << ", splits_per_cluster=" << params.splits_per_cluster
        << ". Increase database_size or decrease num_clusters/n.";
    }

    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params))
        << "embedding_precision (" << params.embedding_precision << "), embedding_length ("
        << params.embedding_length << ") and plaintext_modulus (" << params.plaintext_modulus
        << ") are incompatible for a single split's dot product.";

    CryptoContext ctx = CryptoContext::from_params(params);

    constexpr int kNumIterations = 3; // each iteration does a full eval-key seed+reconstruct,
                                       // a full query seed+reconstruct, plus
                                       // num_clusters * splits_per_cluster * embedding_length
                                       // multiplications and num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- Eval keys: seeded, wire-compressed, reconstructed. NO secret
        // key involved in reconstruction. -------------------------------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- Query: sample embedding values, seed-compress, reconstruct. -----
        std::vector<SignedValue> messages;
        std::vector<int64_t> embedding_values;
        messages.reserve(static_cast<size_t>(params.embedding_length));
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);
            embedding_values.push_back(m.reduced);
        }

        SeededQuery query_wire = build_seeded_query(ctx, secret, embedding_values, params.num_clusters,
                                                     params.desired_cluster_index);
        ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);

        // --- Switch reconstructed embedding ciphertexts to RLWE/eval form,
        // using the RECONSTRUCTED pub. Built once, reused across every
        // cluster AND every split -- same as the non-seeded version. ----------
        std::vector<RLWECTEvalForm> query_eval;
        query_eval.reserve(query.embedding_cts.size());
        for (const auto& lwe_ct : query.embedding_cts) {
            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
            query_eval.emplace_back(rlwe_ct);
        }

        // --- Switch reconstructed selector ciphertexts to RGSW, using the
        // RECONSTRUCTED pub. Built once per cluster, reused across every
        // split of that cluster -- same as the non-seeded version. ------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(query.selector_cts.size());
        for (const auto& gadget_ct : query.selector_cts) {
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Everything below is UNCHANGED from the non-seeded test:
        // per-(cluster,split) score, mask, accumulate; per-split decrypt and
        // verify. -----------------------------------------------------------
        std::vector<std::vector<std::vector<int64_t>>> desired_cluster_raw_values(
            static_cast<size_t>(params.splits_per_cluster));

        std::vector<RLWECT> final_result;
        final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            final_result.emplace_back(ctx.rlwe_param); // zero-initialized accumulator
        }

        for (int64_t c = 0; c < params.num_clusters; ++c) {
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                std::vector<DatabasePolynomialEvalForm> db_split;
                db_split.reserve(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    db_split.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
                }

                if (c == params.desired_cluster_index) {
                    for (int64_t j = 0; j < params.embedding_length; ++j) {
                        desired_cluster_raw_values[static_cast<size_t>(s)].push_back(
                            db_split[static_cast<size_t>(j)].raw_values);
                    }
                }

                RLWECT score = compute_split_score(ctx, query_eval, db_split);

                RLWECT masked(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)].mul(masked, score);

                final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)], masked);
            }
        }

        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            Vector decrypted = secret.rlwe_sk->decrypt_vector(final_result[static_cast<size_t>(s)], ctx.encoding);
            std::vector<int64_t> decoded_signed_vec = decode_to_signed(decrypted, params);
            const auto& raw_values_for_split = desired_cluster_raw_values[static_cast<size_t>(s)];

            for (int64_t i = 0; i < params.n; ++i) {
                int64_t expected = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    int64_t db_reduced_i_j = reduce_mod(
                        raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)],
                        params.plaintext_modulus);
                    expected = (expected + messages[static_cast<size_t>(j)].reduced * db_reduced_i_j) % params.plaintext_modulus;
                }
                EXPECT_EQ(decrypted[i], expected)
                    << "Mod-p mismatch on iteration " << iter << ", split " << s << ", coefficient " << i
                    << ": expected=" << expected << " got=" << decrypted[i];

                int64_t true_sum = 0;
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    true_sum += messages[static_cast<size_t>(j)].raw *
                                raw_values_for_split[static_cast<size_t>(j)][static_cast<size_t>(i)];
                }
                int64_t decoded_signed = decoded_signed_vec[static_cast<size_t>(i)];
                EXPECT_EQ(decoded_signed, true_sum)
                    << "Overflow/corruption on iteration " << iter << ", split " << s << ", coefficient " << i
                    << ": true_sum=" << true_sum << " decoded_signed=" << decoded_signed
                    << " (mod-p residue was " << decrypted[i] << ").";
            }
        }
    }
}
