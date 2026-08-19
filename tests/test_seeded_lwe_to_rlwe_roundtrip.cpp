// test_seeded_lwe_to_rlwe_roundtrip.cpp
//
// Same round trip as test_lwe_to_rlwe_roundtrip.cpp, but exercises the seed
// mechanism (SeededUniformDistribution, see seeded_distribution.hpp) at both
// points it applies:
//
//   1. Eval-key generation: secret.rlwe_sk's distribution is swapped to a
//      fresh seeded one BEFORE generate_client_public_material builds the
//      automorphism keys (LWEToRLWEKeySwitchKey). UNCHANGED by CRT -- one
//      seed, one stream, no dependency on plaintext modulus at all. There's
//      no "reconstruct the eval key from its seed" step here, since that's
//      a separate, bigger piece of future work (reconstructing log2(n)
//      automorphism keys, each itself multiple RLWE ciphertexts).
//
//   2. Query ciphertexts: ONE independent seed/stream PER component ring
//      (params.num_component_rings entries) -- matches SeededQuery's design
//      exactly. Each message is CRT-split (only when r==2 -- for r==1 it's
//      used directly, no split needed at all) and each component encrypted
//      on its own stream. For each ciphertext, ONLY the seed and each
//      component's b value are kept -- simulating "this is all that got
//      sent over the wire" -- and each component ciphertext is rebuilt from
//      scratch via a fresh SeededUniformDistribution built from its OWN
//      seed, consumed in the SAME order originally encrypted.
//
// Parameterized over TWO Params factories, run via the SAME test body --
// same reasoning as test_lwe_to_rlwe_roundtrip.cpp: one implementation,
// exercised under both num_component_rings == 1 and == 2, rather than a
// separate CRT-specific copy that could drift out of sync with this one.
//
// After the LWE->RLWE switch, decryption is just normal RLWE decryption --
// no further seed reconstruction is needed or possible at that point, since
// the switch's automorphism/trace step transforms `a` into something that
// no longer corresponds to any single seed (it's a computed combination of
// the original ciphertext's `a` and the eval key's own `a` components, not
// a fresh uniform sample) -- there's nothing left to reconstruct from a
// seed once you're on the RLWE side.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_seeded_lwe_to_rlwe_roundtrip

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

using namespace FHEDeck;
using namespace psearch;

namespace {

class SeededLweToRlweRoundTrip : public ::testing::TestWithParam<Params (*)()> {};

} // namespace

TEST_P(SeededLweToRlweRoundTrip, EvalKeysAndQueryCiphertextsBothSeeded) {
    Params params = GetParam()();
    CryptoContext ctx = CryptoContext::from_params(params);
    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    constexpr int kNumIterations = 5; // each iteration regenerates keys AND eval keys
    constexpr int kNumCiphertextsPerIteration = 6; // "several" LWE ciphertexts per iteration

    std::mt19937_64 rng(std::random_device{}());
    int64_t r = params.num_component_rings;

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- 1. Eval keys: seed the automorphism-key generation. UNCHANGED
        // by CRT. --------------------------------------------------------------------
        auto eval_key_seed = generate_fresh_seed();
        auto eval_key_dist = std::make_shared<SeededUniformDistribution>(eval_key_seed, 0, params.q);
        secret.rlwe_sk->set_unif_dist(eval_key_dist);

        ClientPublicMaterial pub = generate_client_public_material(ctx, secret); // consumes eval_key_dist

        // --- 2. Query ciphertexts: one independent seed/stream per
        // component ring. -----------------------------------------------------------
        std::vector<std::array<uint8_t, SeededUniformDistribution::kSeedBytes>> query_seeds(static_cast<size_t>(r));
        std::vector<std::shared_ptr<SeededUniformDistribution>> query_dists(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            query_seeds[static_cast<size_t>(ring)] = generate_fresh_seed();
            query_dists[static_cast<size_t>(ring)] =
                std::make_shared<SeededUniformDistribution>(query_seeds[static_cast<size_t>(ring)], 0, params.q);
        }

        std::shared_ptr<Distribution> original_lwe_dist = secret.lwe_sk->set_unif_dist(query_dists[0]);

        std::vector<int64_t> messages; // the ORIGINAL, pre-CRT-split values
        std::vector<std::vector<int64_t>> b_values(static_cast<size_t>(r));   // [ring][k]
        std::vector<std::vector<LWECT>> original_cts(static_cast<size_t>(r)); // [ring][k], sanity check only,
                                                                                 // not used for the actual
                                                                                 // switch/decrypt below
        messages.reserve(kNumCiphertextsPerIteration);
        for (int64_t ring = 0; ring < r; ++ring) {
            b_values[static_cast<size_t>(ring)].reserve(kNumCiphertextsPerIteration);
            original_cts[static_cast<size_t>(ring)].reserve(kNumCiphertextsPerIteration);
        }

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            int64_t m = sample_signed_mod_value(params, rng);
            messages.push_back(m);

            std::vector<int64_t> components;
            if (r == 1) {
                components = {m};
            } else { // r == 2
                auto [c1, c2] = crt_split(m, params.comp_ring_modulus);
                components = {c1, c2};
            }

            for (int64_t ring = 0; ring < r; ++ring) {
                secret.lwe_sk->set_unif_dist(query_dists[static_cast<size_t>(ring)]);
                LWECT ct = secret.lwe_sk->encode_and_encrypt(components[static_cast<size_t>(ring)],
                                                              ctx.component_encodings[static_cast<size_t>(ring)]);
                b_values[static_cast<size_t>(ring)].push_back(ct[0]); // this + query_seeds[ring] is "all that got sent"
                original_cts[static_cast<size_t>(ring)].push_back(ct);
            }
        }

        // Restore the client's normal (non-deterministic) distribution --
        // good practice once a seeded batch is done, not required for this
        // test to pass.
        secret.lwe_sk->set_unif_dist(original_lwe_dist);

        // --- Reconstruct every stream from (seed, b), in the SAME order
        // originally encrypted, and verify. -------------------------------------------
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

                // Sanity check: reconstruction alone (before touching the
                // switch or decryption at all) must reproduce the original
                // ciphertext exactly. If this ever fails, the bug is in the
                // seed/ordering mechanism itself, not downstream.
                EXPECT_TRUE(original_cts[static_cast<size_t>(ring)][static_cast<size_t>(k)].ct_vec() ==
                            reconstructed.ct_vec())
                    << "Reconstruction mismatch on iteration " << iter << ", ring " << ring << ", ciphertext " << k
                    << ": rebuilt LWE ciphertext does not match the original.";

                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, reconstructed); // SAME ksk for every ring

                // Ordinary RLWE decryption from here -- nothing seed-related
                // left to reconstruct, see the file header for why.
                Vector decrypted_vector =
                    secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.component_encodings[static_cast<size_t>(ring)]);
                decrypted_components[static_cast<size_t>(ring)] = static_cast<int64_t>(decrypted_vector[0]);
            }

            int64_t recomposed = (r == 1) ? decrypted_components[0]
                                           : crt_recompose(decrypted_components[0], decrypted_components[1],
                                                            params.comp_ring_modulus);

            EXPECT_EQ(recomposed, messages[static_cast<size_t>(k)])
                << "Mismatch on iteration " << iter << ", ciphertext " << k << " (r=" << r << "): encrypted "
                << messages[static_cast<size_t>(k)] << " but recomposed to " << recomposed;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, SeededLweToRlweRoundTrip,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });