// test_full_scoring_with_splits.cpp
//
// Extends test_full_scoring_with_cluster_selection.cpp with the splits
// dimension: if a cluster holds more documents than fit in one ring's worth
// of n coefficients, it's divided into splits_per_cluster splits, and every
// calculation (per-cluster scoring, RGSW masking) happens independently per
// split. The client ends up with splits_per_cluster final RLWE ciphertexts
// instead of one -- each corresponds to one chunk of documents within
// whichever cluster was selected.
//
// Key structural facts this test exercises:
//   - splits_per_cluster = ceil(cluster_size / n), where
//     cluster_size = database_size / num_clusters (see
//     Params::derive_dependent_parameters). Params::make_test_params() sets
//     the "real" database_size so that cluster_size > n and
//     splits_per_cluster > 1 here; every other test in this suite locally
//     pins database_size = n instead, to stay in the simpler single-split
//     case that's all they need.
//   - The query (query_eval) and the RGSW cluster-selection ciphertexts
//     (rgsw_ct) do NOT depend on split -- a cluster is either selected or
//     not, regardless of which split of it is being scored -- so both are
//     built ONCE per iteration and reused across every split.
//   - Only the database polynomials are split-specific: cluster c, split s
//     has its own l database polynomials, independent of every other
//     (cluster, split) pair.
//   - final_result becomes a vector of splits_per_cluster RLWE ciphertexts,
//     each accumulated by summing masked_{c,s} over all clusters c, for that
//     fixed split s.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_full_scoring_with_splits

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

/// score = sum_j query[j] * db[j], multiplying AND summing entirely in eval
/// form (safe now that PolynomialEvalFormLongInteger::add's missing
/// modular reduction is fixed), with a single eval->coef conversion at the
/// end. Used once per (cluster, split) pair.
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

TEST(FullScoringWithSplits, EachSplitProducesCorrectResultForDesiredCluster) {
    Params params = Params::make_test_params();
    // Uses database_size (and therefore cluster_size / splits_per_cluster)
    // exactly as Params::make_test_params() sets it -- this is the one test
    // in the suite that exercises the "real" (multi-split) database size;
    // every other test locally pins database_size = n to stay in the
    // simpler single-split case.

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

    constexpr int kNumIterations = 3; // each iteration does
                                       // num_clusters * splits_per_cluster * embedding_length
                                       // multiplications plus num_clusters RGSW switches --
                                       // keep this modest.
    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        // --- Query: built once, reused across every cluster AND every
        // split. ---------------------------------------------------------
        std::vector<SignedValue> messages;
        std::vector<RLWECTEvalForm> query_eval;
        messages.reserve(static_cast<size_t>(params.embedding_length));
        query_eval.reserve(static_cast<size_t>(params.embedding_length));

        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            messages.push_back(m);

            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(m.reduced, ctx.encoding);
            RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
            query_eval.emplace_back(rlwe_ct);
        }

        // --- Cluster-selection RGSW ciphertexts: built once per cluster,
        // reused across every split of that cluster. ----------------------
        std::vector<RLWEGadgetCT> rgsw_ct;
        rgsw_ct.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
            rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
        }

        // --- Per (cluster, split): score, mask, accumulate into the
        // matching split's final result. Desired cluster's raw database
        // values are kept per split for verification afterwards. ----------
        // desired_cluster_raw_values[s][j] -> vector<int64_t> of length n
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

        // --- Decrypt and verify EACH split's result independently. -------
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
