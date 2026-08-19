// test_seeded_key_material_and_query_roundtrip.cpp
//
// The full pipeline in one test: eval keys are built with a seed (client
// side), only the wire-compressed form (SeededClientPublicMaterial) is kept,
// and reconstructed into a real, usable ClientPublicMaterial with NO secret
// key involved (server side) -- see seeded_eval_keys.hpp. UNCHANGED by CRT --
// eval keys never depended on the plaintext modulus. Separately, several LWE
// ciphertexts are built with ONE INDEPENDENT SEED PER COMPONENT RING, and
// only (seed, b) is kept for each -- see seeded_distribution.hpp and
// test_seeded_lwe_to_rlwe_roundtrip.cpp for that half of the pattern.
//
// Two checks, using the RECONSTRUCTED public material throughout (never the
// original pub built alongside the wire structs):
//   1. LWE->RLWE: reconstructed LWE ciphertexts (per component ring),
//      switched via the SAME reconstructed lwe_to_rlwe_ksk for every ring,
//      decrypted, recomposed (a no-op when r==1) -- exercises
//      LWEToRLWEKeySwitchKey's reconstruction constructor. Pure round trip,
//      no multiplication/sum involved, so no overflow risk -- the
//      recomposed value is compared directly against the original sampled
//      value (both already canonical non-negative values in the combined
//      space, no centered_residue needed here, unlike check 2).
//   2. LWE'->RGSW: a cluster-selector-style ciphertext (UNCHANGED by CRT --
//      exactly one selector, reused for every ring), switched via the
//      reconstructed lwe_to_rgsw_ksk, then RGSW-multiplied against a random
//      dense RLWE ciphertext (CRT-split into per-ring Vectors) and
//      decrypted -- exercises ct_of_sk_dest's reconstruction specifically,
//      which check 1 never touches. Here centered_residue IS needed on the
//      recomposed value, since the raw database value can be negative --
//      same reasoning as test_rgsw_multiplication.cpp.
//
// Parameterized over TWO Params factories, run via the SAME test body.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_seeded_key_material_and_query_roundtrip

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <vector>

#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

class SeededKeyMaterialAndQueryRoundtrip : public ::testing::TestWithParam<Params (*)()> {};

} // namespace

TEST_P(SeededKeyMaterialAndQueryRoundtrip, ReconstructedEvalKeysSwitchAndDecryptCorrectly) {
    Params params = GetParam()();
    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    constexpr int kNumIterations = 5;
    constexpr int kNumCiphertextsPerIteration = 6; // "several" LWE ciphertexts per iteration

    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- 1. Eval keys: seeded generation, wire-compressed, then
        // reconstructed. UNCHANGED by CRT. -------------------------------------------
        SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, secret);
        ClientPublicMaterial pub = reconstruct_public_material(ctx, params, eval_wire);

        // --- 2. Several LWE (embedding-style) ciphertexts: one independent
        // seed/stream PER COMPONENT RING, kept as (seed, b) only, then
        // reconstructed. --------------------------------------------------------------
        std::vector<std::array<uint8_t, SeededUniformDistribution::kSeedBytes>> query_seeds(static_cast<size_t>(r));
        std::vector<std::shared_ptr<SeededUniformDistribution>> query_dists(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            query_seeds[static_cast<size_t>(ring)] = generate_fresh_seed();
            query_dists[static_cast<size_t>(ring)] =
                std::make_shared<SeededUniformDistribution>(query_seeds[static_cast<size_t>(ring)], 0, params.q);
        }

        std::shared_ptr<Distribution> original_lwe_dist = secret.lwe_sk->set_unif_dist(query_dists[0]);

        std::vector<int64_t> messages; // the ORIGINAL, pre-CRT-split values
        std::vector<std::vector<int64_t>> b_values(static_cast<size_t>(r)); // [ring][k]
        messages.reserve(kNumCiphertextsPerIteration);
        for (int64_t ring = 0; ring < r; ++ring) {
            b_values[static_cast<size_t>(ring)].reserve(kNumCiphertextsPerIteration);
        }

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            // Already reduced mod plaintext_modulus and non-negative, hence
            // also < combined_modulus (derive_dependent_parameters()
            // guarantees combined_modulus >= plaintext_modulus) -- already a
            // valid value to CRT-split directly, no further canonicalization
            // needed.
            int64_t m = sample_signed_mod_value(params, rng);
            messages.push_back(m);

            std::vector<int64_t> components;
            if (r == 1) {
                components = {m};
            } else {
                auto [c1, c2] = crt_split(m, params.comp_ring_modulus);
                components = {c1, c2};
            }

            for (int64_t ring = 0; ring < r; ++ring) {
                secret.lwe_sk->set_unif_dist(query_dists[static_cast<size_t>(ring)]);
                LWECT ct = secret.lwe_sk->encode_and_encrypt(components[static_cast<size_t>(ring)],
                                                              ctx.component_encodings[static_cast<size_t>(ring)]);
                b_values[static_cast<size_t>(ring)].push_back(ct[0]); // this + query_seeds[ring] is "all that got sent"
            }
        }

        secret.lwe_sk->set_unif_dist(original_lwe_dist); // restore normal behaviour

        // --- Check 1: reconstruct each stream, switch via the RECONSTRUCTED
        // lwe_to_rlwe_ksk (SAME for every ring), decrypt, recompose. -----------------
        std::vector<std::shared_ptr<SeededUniformDistribution>> reconstruct_dists(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            reconstruct_dists[static_cast<size_t>(ring)] =
                std::make_shared<SeededUniformDistribution>(query_seeds[static_cast<size_t>(ring)], 0, params.q);
        }

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            std::vector<int64_t> decrypted_components(static_cast<size_t>(r));

            for (int64_t ring = 0; ring < r; ++ring) {
                LWECT reconstructed = reconstruct_lwe_from_seed_and_b(
                    secret.lwe_sk->param(), *reconstruct_dists[static_cast<size_t>(ring)],
                    b_values[static_cast<size_t>(ring)][static_cast<size_t>(k)]);

                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, reconstructed);

                Vector decrypted_vector =
                    secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.component_encodings[static_cast<size_t>(ring)]);
                decrypted_components[static_cast<size_t>(ring)] = static_cast<int64_t>(decrypted_vector[0]);
            }

            // Pure round trip -- no multiplication/sum, no overflow risk, no
            // sign ambiguity (both sides already canonical non-negative) --
            // direct comparison, unlike check 2 below.
            int64_t recomposed = (r == 1) ? decrypted_components[0]
                                           : crt_recompose(decrypted_components[0], decrypted_components[1],
                                                            params.comp_ring_modulus);

            EXPECT_EQ(recomposed, messages[static_cast<size_t>(k)])
                << "LWE->RLWE mismatch on iteration " << iter << ", ciphertext " << k << " (r=" << r
                << "): encrypted " << messages[static_cast<size_t>(k)] << " but recomposed to " << recomposed;
        }

        // --- Check 2: a selector-style LWE' ciphertext, switched via the
        // RECONSTRUCTED lwe_to_rgsw_ksk (exercises ct_of_sk_dest's
        // reconstruction specifically). UNCHANGED by CRT -- exactly ONE
        // selector, reused for every ring below -- then RGSW-multiplied
        // against a random dense RLWE ciphertext (CRT-split into per-ring
        // Vectors) and decrypted. -------------------------------------------------------
        int64_t bit = 1; // arbitrary; correctness doesn't depend on which value
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        RLWEGadgetCT rgsw_ct = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);

        std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
        for (int64_t i = 0; i < params.n; ++i) {
            raw_values[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
        }
        std::vector<Vector> message_vecs = crt_split_vector_from_raw_values(params, raw_values);

        std::vector<Vector> decrypted_per_ring;
        decrypted_per_ring.reserve(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            RLWECT rlwe_ct = secret.rlwe_sk->encode_and_encrypt(message_vecs[static_cast<size_t>(ring)],
                                                                 ctx.component_encodings[static_cast<size_t>(ring)]);

            RLWECT product_ct(ctx.rlwe_param);
            rgsw_ct.mul(product_ct, rlwe_ct); // SAME rgsw_ct for every ring

            decrypted_per_ring.push_back(
                secret.rlwe_sk->decrypt_vector(product_ct, ctx.component_encodings[static_cast<size_t>(ring)]));
        }

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t recomposed = (r == 1) ? decrypted_per_ring[0][i]
                                           : crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i],
                                                            params.comp_ring_modulus);

            // centered_residue IS required here -- expected can be negative
            // (raw_values[i] is signed) -- unlike check 1 above.
            int64_t decoded_signed = centered_residue(recomposed, combined_modulus);
            int64_t expected = (bit == 1) ? raw_values[static_cast<size_t>(i)] : 0;

            EXPECT_EQ(decoded_signed, expected)
                << "LWE'->RGSW mismatch on iteration " << iter << ", coefficient " << i << " (r=" << r
                << "): expected=" << expected << " got=" << decoded_signed;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, SeededKeyMaterialAndQueryRoundtrip,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });