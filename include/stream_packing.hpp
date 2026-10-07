#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "fhe_deck.h"
#include "params.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"

namespace psearch {

// ============================================================================
// Bit-packed byte streams for everything that crosses the network.
//
// Every coefficient is reduced to its canonical residue in [0, q) and written
// with exactly coefficient_bit_width(q) = ceil(log2 q) bits, instead of as a
// full 64-bit int64_t. Seeds are written as raw 16-byte blocks.
//
// The streams carry NO length prefixes and NO parameter metadata: every
// count (n, l, C, r, s, digits, automorphism levels) is derived from the
// public Params on both ends. Unpacking checks that the stream has exactly
// the expected byte length and that every value is < q, and throws
// std::runtime_error otherwise.
//
// Bit order is explicit (little-endian bit stream, built with shifts), so the
// format does not depend on the host's byte order.
//
// Stream layouts (each stream is padded with zero bits to a whole byte):
//
//   Eval keys (one stream, one-time):
//     [eval_key_seed: 16 B]
//     [automorphism b's: log2(n) levels x d_ksk digits x n coefficients]
//     [RGSW message-row b's:    d_prime digits x n coefficients]
//     [RGSW message*sk-row b's: d_prime digits x n coefficients]
//
//   Query (one stream PER COMPONENT RING k, per query):
//     [seed_k: 16 B][l embedding b's of ring k]
//     stream 0 additionally ends with
//     [C x d_prime selector b's]   (the selector continues seeds[0]'s stream,
//                                   see seeded_query.hpp)
//
//   Response (one stream PER COMPONENT RING k, per query):
//     s ciphertexts, each [n coefficients of a][n coefficients of b]
// ============================================================================

using ByteStream = std::vector<uint8_t>;

/// ceil(log2(q)): the number of bits needed for any value in [0, q).
int coefficient_bit_width(int64_t q);

/// All the counts the stream layouts depend on, derived from Params only.
struct WireShape {
    int bits = 0;                        ///< bits per coefficient, ceil(log2 q)
    int64_t n = 0;                       ///< ring dimension
    int64_t embedding_length = 0;        ///< l
    int64_t num_clusters = 0;            ///< C
    int64_t num_component_rings = 0;     ///< r
    int64_t splits_per_cluster = 0;      ///< s
    int64_t automorphism_levels = 0;     ///< log2(n)
    int64_t ksk_digits = 0;              ///< ceil(log_{B_ksk} q)
    int64_t prime_digits = 0;            ///< d' = ceil(log_{B_prime} q)
};

WireShape wire_shape(const Params& params);

/// Exact stream sizes in bytes, as produced by the pack_* functions below.
std::size_t public_material_stream_bytes(const Params& params);
std::size_t query_stream_bytes(const Params& params, int64_t ring);
std::size_t response_stream_bytes(const Params& params);

// --- Low-level bit writer / reader ------------------------------------------

class BitPacker {
public:
    explicit BitPacker(std::size_t reserve_bytes = 0) { buf_.reserve(reserve_bytes); }

    /// Appends the low `width` bits of `value`. Requires value < 2^width,
    /// 1 <= width <= 63.
    void put(uint64_t value, int width);

    /// Appends raw bytes. Only valid on a byte boundary (i.e. before any
    /// put() or directly after finish()-style alignment) -- used for seeds,
    /// which always come first in a stream.
    void put_bytes(const uint8_t* data, std::size_t len);

    /// Flushes any partial byte (zero-padded) and returns the stream.
    ByteStream finish();

private:
    ByteStream buf_;
    unsigned __int128 acc_ = 0;
    int nbits_ = 0;
};

class BitUnpacker {
public:
    explicit BitUnpacker(const ByteStream& buf) : buf_(buf) {}

    uint64_t get(int width);
    void get_bytes(uint8_t* out, std::size_t len);

    /// Throws unless every byte was consumed and the padding bits are zero.
    void expect_end() const;

private:
    const ByteStream& buf_;
    std::size_t pos_ = 0;
    unsigned __int128 acc_ = 0;
    int nbits_ = 0;
};

// --- Eval keys (one-time) ---------------------------------------------------

ByteStream pack_public_material(const Params& params, const SeededClientPublicMaterial& material);
SeededClientPublicMaterial unpack_public_material(const Params& params, const ByteStream& stream);

// --- Query (client -> server), one stream per component ring ----------------

ByteStream pack_query_ring(const Params& params, const SeededQuery& query, int64_t ring);
std::vector<ByteStream> pack_query(const Params& params, const SeededQuery& query); // [ring]

/// Needs all r streams, since the selector lives in stream 0.
SeededQuery unpack_query(const Params& params, const std::vector<ByteStream>& streams);

// --- Response (server -> client), one stream per component ring -------------

/// `ring_cts` = the s response ciphertexts of ONE component ring.
ByteStream pack_response_ring(const Params& params, const std::vector<FHEDeck::RLWECT>& ring_cts);
std::vector<FHEDeck::RLWECT> unpack_response_ring(const CryptoContext& ctx, const Params& params,
                                                  const ByteStream& stream);

/// `cts` laid out [ring][split]; returns one stream per ring.
std::vector<ByteStream> pack_response(const Params& params, const std::vector<std::vector<FHEDeck::RLWECT>>& cts);
std::vector<std::vector<FHEDeck::RLWECT>> unpack_response(const CryptoContext& ctx, const Params& params,
                                                          const std::vector<ByteStream>& streams);

/// Total number of bytes over a set of streams.
std::size_t total_bytes(const std::vector<ByteStream>& streams);

} // namespace psearch
