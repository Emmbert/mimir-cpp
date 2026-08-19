// test_lwe_to_rlwe_roundtrip.cpp
//
// Randomized correctness test: many times, generate a fresh set of keys,
// encrypt a random signed value from the embedding value range (see
// db_polynomial.hpp: [-2^(embedding_precision-1), 2^(embedding_precision-1)-1],
// reduced to its canonical non-negative residue mod plaintext_modulus) with
// LWE, key-switch it to RLWE, decrypt, and verify the result matches the
// original message.
//
// Parameterized over TWO Params factories, run via the SAME test body:
//   - Params::make_test_params()                  (num_component_rings == 1)
//   - Params::make_test_params_component_rings()  (num_component_rings == 2)
// The test body handles both uniformly -- NOT two separate code paths for
// "CRT" vs "non-CRT" -- it loops over params.num_component_rings component(s),
// CRT-splitting the sampled message only when there's more than one ring to
// split it across. For num_component_rings == 1 this loop just runs once,
// with the message used directly (no split needed) -- so there's exactly
// ONE implementation of "encrypt, switch, decrypt, verify" here, exercised
// under both configurations, rather than one for each. This is the same
// reasoning that led to genericizing SeededQuery/reconstruct_query over
// num_component_rings rather than writing a separate CRT variant.
//
// Crypto material is split into three categories (see key_material.hpp):
//   - global: psearch::CryptoContext (rlwe_param, gadgets, encoding,
//     component_encodings) -- built once from Params, same for every client.
//   - secret: ClientSecretMaterial -- never leaves the client. Regenerated
//     fresh every iteration here, since we're stress-testing keygen too.
//     UNCHANGED by CRT -- key material never depended on the plaintext
//     modulus, only on q/n/the secret key.
//   - public: ClientPublicMaterial -- what would actually be sent to the
//     server. The SAME lwe_to_rlwe_ksk is reused across every component
//     ring -- confirmed earlier in this project that key switching doesn't
//     depend on which plaintext modulus a ciphertext was encrypted under.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_lwe_to_rlwe_roundtrip
//   ./test_lwe_to_rlwe_roundtrip --gtest_filter=LweToRlweRoundTrip.*

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

class LweToRlweRoundTrip : public ::testing::TestWithParam<Params (*)()> {};

} // namespace

TEST_P(LweToRlweRoundTrip, ManyRandomMessagesFreshKeysEachTime) {
    Params params = GetParam()();
    CryptoContext ctx = CryptoContext::from_params(params); // global, built once

    ASSERT_EQ(static_cast<int64_t>(ctx.component_encodings.size()), params.num_component_rings);

    constexpr int kNumIterations = 10;

    std::mt19937_64 rng(std::random_device{}());
    int64_t r = params.num_component_rings;

    for (int iter = 0; iter < kNumIterations; ++iter) {
        std::cout << "i = " << iter << std::endl;
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        int64_t message = sample_signed_mod_value(params, rng);

        // r == 1: components = {message}, no CRT involved at all.
        // r == 2: components = {message mod p1, message mod p2}.
        std::vector<int64_t> components;
        if (r == 1) {
            components = {message};
        } else {
            auto [c1, c2] = crt_split(message, params.comp_ring_modulus);
            components = {c1, c2};
        }

        std::vector<int64_t> decrypted_components(static_cast<size_t>(r));
        for (int64_t k = 0; k < r; ++k) {
            std::cout << "encodign for component ring = " << k << std::endl;
            const PlaintextEncoding& encoding = ctx.component_encodings[static_cast<size_t>(k)];

            LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(components[static_cast<size_t>(k)], encoding);

            RLWECT rlwe_ct(ctx.rlwe_param);
            pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct); // SAME ksk for every ring

            Vector decrypted_vector = secret.rlwe_sk->decrypt_vector(rlwe_ct, encoding);
            decrypted_components[static_cast<size_t>(k)] = static_cast<int64_t>(decrypted_vector[0]);

            EXPECT_EQ(decrypted_components[static_cast<size_t>(k)], components[static_cast<size_t>(k)])
                << "Component " << k << " alone did not round-trip correctly on iteration " << iter
                << " (r=" << r << ").";
        }

        int64_t recomposed = (r == 1) ? decrypted_components[0]
                                       : crt_recompose(decrypted_components[0], decrypted_components[1],
                                                        params.comp_ring_modulus);

        EXPECT_EQ(recomposed, message)
            << "Mismatch on iteration " << iter << " (r=" << r << "): encrypted " << message
            << " but recomposed to " << recomposed;
    }
}

INSTANTIATE_TEST_SUITE_P(SingleAndTwoComponentRings, LweToRlweRoundTrip,
                          ::testing::Values(&Params::make_test_params, &Params::make_test_params_component_rings),
                          [](const ::testing::TestParamInfo<Params (*)()>& info) {
                              return info.param == &Params::make_test_params ? "SingleRing" : "TwoComponentRings";
                          });