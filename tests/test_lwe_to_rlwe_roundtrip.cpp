// test_lwe_to_rlwe_roundtrip.cpp
//
// Randomized correctness test: many times, generate a fresh set of keys,
// encrypt a random signed value from the embedding value range (see
// db_polynomial.hpp: [-2^(embedding_precision-1), 2^(embedding_precision-1)-1],
// reduced to its canonical non-negative residue mod plaintext_modulus) with
// LWE, key-switch it to RLWE, decrypt, and verify the result matches the
// original message.
//
// Crypto material is split into three categories (see key_material.hpp):
//   - global: psearch::CryptoContext (rlwe_param, gadget, encoding) — built
//     once from Params, same for every client.
//   - secret: ClientSecretMaterial — never leaves the client. Regenerated
//     fresh every iteration here, since we're stress-testing keygen too.
//   - public: ClientPublicMaterial — what would actually be sent to the
//     server. Only this, plus the global context, is used by switch_to_rlwe;
//     it never touches the secret material.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_lwe_to_rlwe_roundtrip
//   ./test_lwe_to_rlwe_roundtrip --gtest_filter=LweToRlweRoundTrip.*

#include <gtest/gtest.h>

#include <random>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

/// Encrypts `message` as an LWE ciphertext under the client's secret key,
/// scaled by ctx.encoding so it pairs correctly with decrypt_rlwe's
/// encoding-aware decode. Only touches secret material, as encryption should.
LWECT encrypt_lwe(const CryptoContext& ctx, const ClientSecretMaterial& secret, int64_t message) {
    return secret.lwe_sk->encode_and_encrypt(message, ctx.encoding);
}

/// Key-switches an LWE ciphertext into an RLWE ciphertext. Deliberately takes
/// only the *public* material — this is exactly the operation the server
/// performs, and the server never has access to ClientSecretMaterial.
RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

/// Decrypts an RLWE ciphertext, reading the message out of coefficient 0.
/// Back to secret material only — decryption is a client-side operation.
int64_t decrypt_rlwe(const CryptoContext& ctx, const ClientSecretMaterial& secret, const RLWECT& rlwe_ct) {
    Vector decrypted_vector = secret.rlwe_sk->decrypt_vector(rlwe_ct, ctx.encoding);
    return static_cast<int64_t>(decrypted_vector[0]);
}

} // namespace

TEST(LweToRlweRoundTrip, ManyRandomMessagesFreshKeysEachTime) {
    Params params = Params::make_test_params();
    CryptoContext ctx = CryptoContext::from_params(params); // global, built once

    // TODO: tune. Each iteration regenerates a full key set (RLWESK keygen +
    // key-switching key generation), which is the slow part — if this test
    // gets too slow, consider a second variant that reuses one
    // ClientSecretMaterial/ClientPublicMaterial pair across many random
    // messages, to separate "does keygen ever produce a bad key" from "does
    // encrypt/switch/decrypt work for many messages".
    constexpr int kNumIterations = 10;

    std::mt19937_64 rng(std::random_device{}());

    for (int iter = 0; iter < kNumIterations; ++iter) {
        ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
        ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

        int64_t message = sample_signed_mod_value(params, rng);

        LWECT lwe_ct = encrypt_lwe(ctx, secret, message);
        RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct); // note: no `secret` in scope here
        int64_t decrypted = decrypt_rlwe(ctx, secret, rlwe_ct);

        EXPECT_EQ(decrypted, message)
            << "Mismatch on iteration " << iter << ": encrypted " << message
            << " but decrypted to " << decrypted;
    }
}