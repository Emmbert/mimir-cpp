// test_seeded_lwe_to_rlwe_roundtrip.cpp
//
// Same round trip as test_lwe_to_rlwe_roundtrip.cpp, but exercises the seed
// mechanism (SeededUniformDistribution, see seeded_distribution.hpp) at both
// points it applies:
//
//   1. Eval-key generation: secret.rlwe_sk's distribution is swapped to a
//      fresh seeded one BEFORE generate_client_public_material builds the
//      automorphism keys (LWEToRLWEKeySwitchKey). This just checks that
//      generating eval keys with seeded randomness still produces a working
//      key -- there's no "reconstruct the eval key from its seed" step here,
//      since that's a separate, bigger piece of future work (reconstructing
//      log2(n) automorphism keys, each itself multiple RLWE ciphertexts).
//
//   2. Query ciphertexts: secret.lwe_sk's distribution is swapped to a
//      SECOND, independent seeded one before encrypting several messages.
//      For each ciphertext, ONLY the seed and each ciphertext's b value are
//      kept -- simulating "this is all that got sent over the wire" -- and
//      the full ciphertext (specifically its `a` component) is rebuilt from
//      scratch via a fresh SeededUniformDistribution built from the SAME
//      seed, consumed in the SAME order the ciphertexts were originally
//      encrypted in (one continuous stream, matching how the seed mechanism
//      is meant to be used -- see seeded_distribution.hpp).
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

#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"

using namespace FHEDeck;
using namespace psearch;


TEST(SeededLweToRlweRoundTrip, EvalKeysAndQueryCiphertextsBothSeeded) {
    Params params = Params::make_test_params();
    CryptoContext ctx = CryptoContext::from_params(params); // global, built once

    constexpr int kNumIterations = 5; // each iteration regenerates keys AND eval keys
    constexpr int kNumCiphertextsPerIteration = 6; // "several" LWE ciphertexts per iteration

    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

        // --- 1. Eval keys: seed the automorphism-key generation. -----------
        auto eval_key_seed = generate_fresh_seed();
        auto eval_key_dist = std::make_shared<SeededUniformDistribution>(eval_key_seed, 0, params.q);
        secret.rlwe_sk->set_unif_dist(eval_key_dist);

        ClientPublicMaterial pub = generate_client_public_material(ctx, secret); // consumes eval_key_dist

        // --- 2. Query ciphertexts: seed the encryption, keep only (seed, b). ---
        auto query_seed = generate_fresh_seed();
        auto query_dist = std::make_shared<SeededUniformDistribution>(query_seed, 0, params.q);
        std::shared_ptr<Distribution> original_lwe_dist = secret.lwe_sk->set_unif_dist(query_dist);

        std::vector<int64_t> messages;
        std::vector<int64_t> b_values;
        std::vector<LWECT> original_cts; // kept ONLY for the reconstruction sanity check below,
                                          // not used for the actual switch/decrypt -- see the
                                          // EXPECT_EQ on ct_vec() further down.
        messages.reserve(kNumCiphertextsPerIteration);
        b_values.reserve(kNumCiphertextsPerIteration);
        original_cts.reserve(kNumCiphertextsPerIteration);

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            int64_t m = sample_signed_mod_value(params, rng);
            messages.push_back(m);

            LWECT ct = secret.lwe_sk->encode_and_encrypt(m, ctx.encoding);
            b_values.push_back(ct[0]); // this + query_seed is "all that got sent"
            original_cts.push_back(ct);
        }

        // Restore the client's normal (non-deterministic) distribution --
        // good practice once a seeded batch is done, not required for this
        // test to pass.
        secret.lwe_sk->set_unif_dist(original_lwe_dist);

        // --- Reconstruct each ciphertext from (query_seed, b), in the SAME
        // order they were originally encrypted, using one continuing stream. ---
        auto reconstruct_dist = std::make_shared<SeededUniformDistribution>(query_seed, 0, params.q);

        for (int k = 0; k < kNumCiphertextsPerIteration; ++k) {
            LWECT reconstructed =
                reconstruct_lwe_from_seed_and_b(secret.lwe_sk->param(), *reconstruct_dist, b_values[k]);

            // Sanity check: reconstruction alone (before touching the switch
            // or decryption at all) must reproduce the original ciphertext
            // exactly. If this ever fails, the bug is in the seed/ordering
            // mechanism itself, not downstream.
            EXPECT_TRUE(original_cts[static_cast<size_t>(k)].ct_vec() == reconstructed.ct_vec())
                << "Reconstruction mismatch on iteration " << iter << ", ciphertext " << k
                << ": rebuilt LWE ciphertext does not match the original.";

            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, reconstructed);

            // Ordinary RLWE decryption from here -- nothing seed-related left
            // to reconstruct, see the file header for why.
            Vector decrypted_vector = secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.encoding);
            int64_t decrypted = static_cast<int64_t>(decrypted_vector[0]);

            EXPECT_EQ(decrypted, messages[static_cast<size_t>(k)])
                << "Mismatch on iteration " << iter << ", ciphertext " << k << ": encrypted "
                << messages[static_cast<size_t>(k)] << " but decrypted to " << decrypted;
        }
    }
}
