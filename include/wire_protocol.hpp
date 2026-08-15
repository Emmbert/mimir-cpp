#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "fhe_deck.h"
#include "params.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"

namespace psearch {

/// A client-generated 128-bit session token, sent with every Registration
/// and Query message so the server can look up (or store) that client's
/// unpacked eval keys. Generated the same way as any other seed
/// (generate_fresh_seed) -- doesn't need to be secret, just unpredictable
/// enough that one client can't guess another's session ID.
constexpr size_t kSessionIdBytes = SeededUniformDistribution::kSeedBytes;
using SessionId = std::array<uint8_t, kSessionIdBytes>;

/// What the server sends back: one RLWE ciphertext per split. Serialized as
/// raw a/b coefficients (NOT any FHE-Deck cereal format -- USE_CEREAL isn't
/// confirmed enabled in this build, so this avoids depending on it).
struct QueryResponse {
    std::vector<FHEDeck::RLWECT> ciphertexts; // length splits_per_cluster
};

/// Everything below writes into / reads from a plain byte buffer.
/// Fixed-width fields (int64_t, uint32_t length prefixes) are written in the
/// host's native byte order -- correct for same-architecture client/server
/// (the "run locally first" case, and in practice any normal x86_64/ARM64
/// deployment), but NOT portable across differing-endianness machines. If
/// that ever becomes a real deployment scenario, these need explicit
/// htole64/le64toh-style normalization before they're safe to use.

std::vector<uint8_t> serialize_seeded_query(const SeededQuery& query);
SeededQuery deserialize_seeded_query(const std::vector<uint8_t>& buf);

std::vector<uint8_t> serialize_seeded_public_material(const SeededClientPublicMaterial& material);
SeededClientPublicMaterial deserialize_seeded_public_material(const std::vector<uint8_t>& buf);

/// Needs `ctx` on both ends to know n/q -- the wire format itself carries no
/// parameter metadata, matching how every other seeded reconstruction
/// function in this codebase works (params/ctx passed explicitly, never
/// embedded in the wire data).
std::vector<uint8_t> serialize_query_response(const CryptoContext& ctx, const QueryResponse& response);
QueryResponse deserialize_query_response(const CryptoContext& ctx, const std::vector<uint8_t>& buf);

// --- Session-wrapped envelopes: [session_id][the struct above], as sent
// over the wire for MessageType::Registration and MessageType::Query. -------

struct RegistrationMessage {
    SessionId session_id;
    SeededClientPublicMaterial material;
};
std::vector<uint8_t> serialize_registration_message(const RegistrationMessage& msg);
RegistrationMessage deserialize_registration_message(const std::vector<uint8_t>& buf);

struct QueryMessage {
    SessionId session_id;
    SeededQuery query;
};
std::vector<uint8_t> serialize_query_message(const QueryMessage& msg);
QueryMessage deserialize_query_message(const std::vector<uint8_t>& buf);

} // namespace psearch
