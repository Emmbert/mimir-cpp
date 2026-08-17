#pragma once
#include <cstdint>
#include <vector>

#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "wire_protocol.hpp" // SessionId

namespace psearch {

/// Everything the client needs to remember BETWEEN registration and a
/// query: its own secret key material and session ID. Deliberately holds
/// nothing about the network transport -- this type, and every function
/// below, is meant to be usable unchanged from a native CLI now, and later
/// from a different entry point entirely (e.g. compiled to WASM and called
/// from browser JS, with the actual network transport handled on the JS
/// side via fetch/WebSocket instead of ever touching a socket here).
struct ClientSession {
    SessionId session_id;
    ClientSecretMaterial secret;
};

struct RegistrationBundle {
    ClientSession session;      // keep this -- needed for build_query_message and decrypt_and_find_best later
    std::vector<uint8_t> message_bytes; // send this as a MessageType::Registration payload
};

/// Generates fresh keys and a fresh session ID, builds the seeded eval-key
/// wire form, and returns both the session state to keep and the exact
/// bytes to send.
RegistrationBundle build_registration(const CryptoContext& ctx, const Params& params);

/// Builds the exact bytes to send as a MessageType::Query payload, for an
/// already-registered session. `session` is non-const: building a seeded
/// query swaps session.secret.lwe_sk's distribution for the duration (see
/// seeded_query.hpp).
std::vector<uint8_t> build_query_message(const CryptoContext& ctx, ClientSession& session, const Params& params,
                                          const std::vector<int64_t>& embedding_values,
                                          int64_t desired_cluster_index);

struct ClientQueryResult {
    int64_t split_index;
    int64_t position_in_split;
    int64_t score; // the decoded signed score at that position
};

/// Decrypts a QueryResponse's raw bytes (exactly what a Query response's
/// payload contains) using this session's secret key, and returns the
/// position of the largest score found across every split.
ClientQueryResult decrypt_and_find_best(const CryptoContext& ctx, const ClientSession& session, const Params& params,
                                         const std::vector<uint8_t>& response_bytes);

} // namespace psearch
