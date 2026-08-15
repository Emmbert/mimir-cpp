// test_wire_protocol_roundtrip.cpp
//
// Round-trip tests for every serialize/deserialize pair in wire_protocol.hpp:
// construct a struct with known, non-trivial values -- DIFFERENT sizes at
// each nesting level, deliberately, since a test where every level happens
// to be the same length could hide an indexing/ordering bug that only shows
// up with real, uneven data -- serialize, deserialize, and check every
// field survives exactly.
//
// This is the ONLY thing in the test suite that directly exercises
// wire_protocol.cpp, even though it's exactly what client.cpp/server.cpp
// actually send over the network.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_wire_protocol_roundtrip

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "params.hpp"
#include "seeded_distribution.hpp"
#include "wire_protocol.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

SessionId make_session_id(uint8_t fill_start) {
    SessionId id;
    for (size_t i = 0; i < id.size(); ++i) {
        id[i] = static_cast<uint8_t>(fill_start + i);
    }
    return id;
}

} // namespace

TEST(WireProtocolRoundtrip, SeededQuery) {
    SeededQuery query;
    query.seed = make_session_id(1);
    query.embedding_b_values = {10, -20, 30, -40, 5};
    // Different-length inner vectors per cluster, deliberately.
    query.selector_b_values = {
        {1, 2, 3},
        {4, 5, 6, 7, 8},
        {9},
    };

    auto bytes = serialize_seeded_query(query);
    SeededQuery decoded = deserialize_seeded_query(bytes);

    EXPECT_EQ(decoded.seed, query.seed);
    EXPECT_EQ(decoded.embedding_b_values, query.embedding_b_values);
    EXPECT_EQ(decoded.selector_b_values, query.selector_b_values);
}

TEST(WireProtocolRoundtrip, SeededClientPublicMaterial) {
    SeededClientPublicMaterial material;
    material.eval_key_seed = make_session_id(50);
    // Three levels of nesting, different sizes at every level -- this is
    // the structure most likely to hide a bug if any level's loop bound is
    // wrong.
    material.automorphism_b_values = {
        {{1, 2}, {3, 4}, {5, 6}},  // level 0: 3 digits, 2 coefficients each
        {{7, 8, 9}, {10, 11, 12}}, // level 1: 2 digits, 3 coefficients each
    };
    material.rgsw_message_row_b_values = {{-1, -2}, {-3, -4}, {-5, -6}};
    material.rgsw_message_sk_row_b_values = {{100, 200}, {300, 400}};

    auto bytes = serialize_seeded_public_material(material);
    SeededClientPublicMaterial decoded = deserialize_seeded_public_material(bytes);

    EXPECT_EQ(decoded.eval_key_seed, material.eval_key_seed);
    EXPECT_EQ(decoded.automorphism_b_values, material.automorphism_b_values);
    EXPECT_EQ(decoded.rgsw_message_row_b_values, material.rgsw_message_row_b_values);
    EXPECT_EQ(decoded.rgsw_message_sk_row_b_values, material.rgsw_message_sk_row_b_values);
}

TEST(WireProtocolRoundtrip, QueryResponse) {
    Params params = Params::make_test_params();
    CryptoContext ctx = CryptoContext::from_params(params);
    std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> dist(0, params.q - 1);

    QueryResponse response;
    for (int i = 0; i < 3; ++i) {
        Polynomial a(params.n, params.q);
        Polynomial b(params.n, params.q);
        for (int64_t j = 0; j < params.n; ++j) {
            a[j] = dist(rng);
            b[j] = dist(rng);
        }
        response.ciphertexts.emplace_back(ctx.rlwe_param, a, b);
    }

    auto bytes = serialize_query_response(ctx, response);
    QueryResponse decoded = deserialize_query_response(ctx, bytes);

    ASSERT_EQ(decoded.ciphertexts.size(), response.ciphertexts.size());
    for (size_t k = 0; k < response.ciphertexts.size(); ++k) {
        for (int64_t j = 0; j < params.n; ++j) {
            EXPECT_EQ(decoded.ciphertexts[k].a()[j], response.ciphertexts[k].a()[j])
                << "ciphertext " << k << ", a[" << j << "]";
            EXPECT_EQ(decoded.ciphertexts[k].b()[j], response.ciphertexts[k].b()[j])
                << "ciphertext " << k << ", b[" << j << "]";
        }
    }
}

TEST(WireProtocolRoundtrip, RegistrationMessageEnvelope) {
    RegistrationMessage msg;
    msg.session_id = make_session_id(7);
    msg.material.eval_key_seed = make_session_id(90);
    msg.material.automorphism_b_values = {{{1, 2}, {3, 4}}};
    msg.material.rgsw_message_row_b_values = {{5, 6}};
    msg.material.rgsw_message_sk_row_b_values = {{7, 8}};

    auto bytes = serialize_registration_message(msg);
    RegistrationMessage decoded = deserialize_registration_message(bytes);

    EXPECT_EQ(decoded.session_id, msg.session_id);
    EXPECT_EQ(decoded.material.eval_key_seed, msg.material.eval_key_seed);
    EXPECT_EQ(decoded.material.automorphism_b_values, msg.material.automorphism_b_values);
    EXPECT_EQ(decoded.material.rgsw_message_row_b_values, msg.material.rgsw_message_row_b_values);
    EXPECT_EQ(decoded.material.rgsw_message_sk_row_b_values, msg.material.rgsw_message_sk_row_b_values);
}

TEST(WireProtocolRoundtrip, QueryMessageEnvelope) {
    QueryMessage msg;
    msg.session_id = make_session_id(15);
    msg.query.seed = make_session_id(200);
    msg.query.embedding_b_values = {1, -1, 2, -2};
    msg.query.selector_b_values = {{1, 2}, {3, 4, 5}};

    auto bytes = serialize_query_message(msg);
    QueryMessage decoded = deserialize_query_message(bytes);

    EXPECT_EQ(decoded.session_id, msg.session_id);
    EXPECT_EQ(decoded.query.seed, msg.query.seed);
    EXPECT_EQ(decoded.query.embedding_b_values, msg.query.embedding_b_values);
    EXPECT_EQ(decoded.query.selector_b_values, msg.query.selector_b_values);
}

TEST(WireProtocolRoundtrip, TruncatedBufferThrowsRatherThanReadingGarbage) {
    SeededQuery query;
    query.seed = make_session_id(1);
    query.embedding_b_values = {1, 2, 3};
    query.selector_b_values = {{4, 5}};

    auto bytes = serialize_seeded_query(query);
    ASSERT_GT(bytes.size(), 4u);
    bytes.resize(bytes.size() - 4); // truncate -- simulates a dropped/corrupted message

    EXPECT_THROW(deserialize_seeded_query(bytes), std::runtime_error);
}
