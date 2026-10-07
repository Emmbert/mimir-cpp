#include "stream_packing.hpp"

#include <stdexcept>
#include <string>

#include "seeded_distribution.hpp"

using namespace FHEDeck;

namespace psearch {

namespace {

constexpr std::size_t kSeedBytes = SeededUniformDistribution::kSeedBytes;

/// Canonical residue in [0, q). The in-memory representation is a signed
/// int64_t and is not guaranteed to be canonical, so this is applied to
/// every value before it is truncated to `bits` bits.
inline uint64_t canonical(int64_t v, int64_t q) {
    int64_t m = v % q;
    if (m < 0) {
        m += q;
    }
    return static_cast<uint64_t>(m);
}

/// Reads one coefficient and checks it is a valid residue.
inline int64_t read_coefficient(BitUnpacker& in, int bits, int64_t q) {
    uint64_t v = in.get(bits);
    if (v >= static_cast<uint64_t>(q)) {
        throw std::runtime_error("stream_packing: coefficient " + std::to_string(v) + " >= q (" +
                                 std::to_string(q) + ") -- stream corrupted or packed with different parameters");
    }
    return static_cast<int64_t>(v);
}

inline std::size_t bits_to_bytes(uint64_t bits) { return static_cast<std::size_t>((bits + 7) / 8); }

void check_size(const ByteStream& stream, std::size_t expected, const char* what) {
    if (stream.size() != expected) {
        throw std::runtime_error(std::string("stream_packing: ") + what + " stream has " +
                                 std::to_string(stream.size()) + " bytes, expected " + std::to_string(expected));
    }
}

void write_coefficients(BitPacker& out, const std::vector<int64_t>& values, std::size_t expected_count, int bits,
                        int64_t q, const char* what) {
    if (values.size() != expected_count) {
        throw std::invalid_argument(std::string("stream_packing: ") + what + " has " +
                                    std::to_string(values.size()) + " values, expected " +
                                    std::to_string(expected_count));
    }
    for (int64_t v : values) {
        out.put(canonical(v, q), bits);
    }
}

std::vector<int64_t> read_coefficients(BitUnpacker& in, std::size_t count, int bits, int64_t q) {
    std::vector<int64_t> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = read_coefficient(in, bits, q);
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------

int coefficient_bit_width(int64_t q) {
    if (q < 2) {
        throw std::invalid_argument("coefficient_bit_width: q must be >= 2");
    }
    // Bit length of (q - 1) == ceil(log2 q).
    int bits = 0;
    for (uint64_t v = static_cast<uint64_t>(q - 1); v != 0; v >>= 1) {
        ++bits;
    }
    return bits;
}

WireShape wire_shape(const Params& params) {
    WireShape w;
    w.bits = coefficient_bit_width(params.q);
    w.n = params.n;
    w.embedding_length = params.embedding_length;
    w.num_clusters = params.num_clusters;
    w.num_component_rings = params.num_component_rings;
    w.splits_per_cluster = params.splits_per_cluster;
    // LWEToRLWEKeySwitchKey::key_switching_key_gen: i = 2, 4, ..., n.
    w.automorphism_levels = 0;
    for (int64_t i = 2; i <= params.n; i *= 2) {
        ++w.automorphism_levels;
    }
    // Same formula SignedDecompositionGadget / LWEGadgetCT use.
    w.ksk_digits = Utils::power_times(params.q, params.decomposition_base_ksk);
    w.prime_digits = Utils::power_times(params.q, params.decomposition_base_prime);
    return w;
}

std::size_t public_material_stream_bytes(const Params& params) {
    WireShape w = wire_shape(params);
    uint64_t coeffs = static_cast<uint64_t>(w.automorphism_levels * w.ksk_digits * w.n + 2 * w.prime_digits * w.n);
    return kSeedBytes + bits_to_bytes(coeffs * static_cast<uint64_t>(w.bits));
}

std::size_t query_stream_bytes(const Params& params, int64_t ring) {
    WireShape w = wire_shape(params);
    uint64_t coeffs = static_cast<uint64_t>(w.embedding_length);
    if (ring == 0) {
        coeffs += static_cast<uint64_t>(w.num_clusters * w.prime_digits);
    }
    return kSeedBytes + bits_to_bytes(coeffs * static_cast<uint64_t>(w.bits));
}

std::size_t response_stream_bytes(const Params& params) {
    WireShape w = wire_shape(params);
    uint64_t coeffs = static_cast<uint64_t>(w.splits_per_cluster * 2 * w.n);
    return bits_to_bytes(coeffs * static_cast<uint64_t>(w.bits));
}

std::size_t total_bytes(const std::vector<ByteStream>& streams) {
    std::size_t sum = 0;
    for (const auto& s : streams) {
        sum += s.size();
    }
    return sum;
}

// --- BitPacker / BitUnpacker ------------------------------------------------

void BitPacker::put(uint64_t value, int width) {
    // nbits_ < 8 on entry and width <= 63, so acc_ never exceeds 71 bits.
    acc_ |= static_cast<unsigned __int128>(value) << nbits_;
    nbits_ += width;
    while (nbits_ >= 8) {
        buf_.push_back(static_cast<uint8_t>(acc_));
        acc_ >>= 8;
        nbits_ -= 8;
    }
}

void BitPacker::put_bytes(const uint8_t* data, std::size_t len) {
    if (nbits_ != 0) {
        throw std::logic_error("BitPacker::put_bytes: not on a byte boundary");
    }
    buf_.insert(buf_.end(), data, data + len);
}

ByteStream BitPacker::finish() {
    if (nbits_ > 0) {
        buf_.push_back(static_cast<uint8_t>(acc_));
    }
    acc_ = 0;
    nbits_ = 0;
    return std::move(buf_);
}

uint64_t BitUnpacker::get(int width) {
    while (nbits_ < width) {
        if (pos_ >= buf_.size()) {
            throw std::runtime_error("BitUnpacker: buffer underrun -- stream truncated or malformed");
        }
        acc_ |= static_cast<unsigned __int128>(buf_[pos_++]) << nbits_;
        nbits_ += 8;
    }
    uint64_t mask = (width == 64) ? ~uint64_t(0) : ((uint64_t(1) << width) - 1);
    uint64_t v = static_cast<uint64_t>(acc_) & mask;
    acc_ >>= width;
    nbits_ -= width;
    return v;
}

void BitUnpacker::get_bytes(uint8_t* out, std::size_t len) {
    if (nbits_ != 0) {
        throw std::logic_error("BitUnpacker::get_bytes: not on a byte boundary");
    }
    if (pos_ + len > buf_.size()) {
        throw std::runtime_error("BitUnpacker: buffer underrun -- stream truncated or malformed");
    }
    for (std::size_t i = 0; i < len; ++i) {
        out[i] = buf_[pos_ + i];
    }
    pos_ += len;
}

void BitUnpacker::expect_end() const {
    if (pos_ != buf_.size()) {
        throw std::runtime_error("BitUnpacker: " + std::to_string(buf_.size() - pos_) + " trailing bytes");
    }
    if (acc_ != 0) {
        throw std::runtime_error("BitUnpacker: non-zero padding bits");
    }
}

// --- Eval keys --------------------------------------------------------------

ByteStream pack_public_material(const Params& params, const SeededClientPublicMaterial& material) {
    WireShape w = wire_shape(params);
    const auto n = static_cast<std::size_t>(w.n);

    if (static_cast<int64_t>(material.automorphism_b_values.size()) != w.automorphism_levels ||
        static_cast<int64_t>(material.rgsw_message_row_b_values.size()) != w.prime_digits ||
        static_cast<int64_t>(material.rgsw_message_sk_row_b_values.size()) != w.prime_digits) {
        throw std::invalid_argument("pack_public_material: material shape does not match params");
    }

    BitPacker out(public_material_stream_bytes(params));
    out.put_bytes(material.eval_key_seed.data(), material.eval_key_seed.size());

    for (const auto& level : material.automorphism_b_values) {
        if (static_cast<int64_t>(level.size()) != w.ksk_digits) {
            throw std::invalid_argument("pack_public_material: automorphism level has wrong digit count");
        }
        for (const auto& digit : level) {
            write_coefficients(out, digit, n, w.bits, params.q, "automorphism key digit");
        }
    }
    for (const auto& digit : material.rgsw_message_row_b_values) {
        write_coefficients(out, digit, n, w.bits, params.q, "RGSW message-row digit");
    }
    for (const auto& digit : material.rgsw_message_sk_row_b_values) {
        write_coefficients(out, digit, n, w.bits, params.q, "RGSW message*sk-row digit");
    }
    return out.finish();
}

SeededClientPublicMaterial unpack_public_material(const Params& params, const ByteStream& stream) {
    check_size(stream, public_material_stream_bytes(params), "eval key");
    WireShape w = wire_shape(params);
    const auto n = static_cast<std::size_t>(w.n);

    BitUnpacker in(stream);
    SeededClientPublicMaterial material;
    in.get_bytes(material.eval_key_seed.data(), material.eval_key_seed.size());

    material.automorphism_b_values.resize(static_cast<std::size_t>(w.automorphism_levels));
    for (auto& level : material.automorphism_b_values) {
        level.reserve(static_cast<std::size_t>(w.ksk_digits));
        for (int64_t d = 0; d < w.ksk_digits; ++d) {
            level.push_back(read_coefficients(in, n, w.bits, params.q));
        }
    }
    material.rgsw_message_row_b_values.reserve(static_cast<std::size_t>(w.prime_digits));
    for (int64_t d = 0; d < w.prime_digits; ++d) {
        material.rgsw_message_row_b_values.push_back(read_coefficients(in, n, w.bits, params.q));
    }
    material.rgsw_message_sk_row_b_values.reserve(static_cast<std::size_t>(w.prime_digits));
    for (int64_t d = 0; d < w.prime_digits; ++d) {
        material.rgsw_message_sk_row_b_values.push_back(read_coefficients(in, n, w.bits, params.q));
    }
    in.expect_end();
    return material;
}

// --- Query ------------------------------------------------------------------

ByteStream pack_query_ring(const Params& params, const SeededQuery& query, int64_t ring) {
    WireShape w = wire_shape(params);
    if (ring < 0 || ring >= w.num_component_rings) {
        throw std::invalid_argument("pack_query_ring: ring index out of range");
    }
    if (static_cast<int64_t>(query.seeds.size()) != w.num_component_rings ||
        static_cast<int64_t>(query.embedding_b_values.size()) != w.num_component_rings) {
        throw std::invalid_argument("pack_query_ring: query has wrong number of component rings");
    }
    const auto k = static_cast<std::size_t>(ring);

    BitPacker out(query_stream_bytes(params, ring));
    out.put_bytes(query.seeds[k].data(), query.seeds[k].size());
    write_coefficients(out, query.embedding_b_values[k], static_cast<std::size_t>(w.embedding_length), w.bits,
                       params.q, "embedding b-values");

    if (ring == 0) {
        if (static_cast<int64_t>(query.selector_b_values.size()) != w.num_clusters) {
            throw std::invalid_argument("pack_query_ring: selector has wrong number of clusters");
        }
        for (const auto& cluster : query.selector_b_values) {
            write_coefficients(out, cluster, static_cast<std::size_t>(w.prime_digits), w.bits, params.q,
                               "selector b-values");
        }
    }
    return out.finish();
}

std::vector<ByteStream> pack_query(const Params& params, const SeededQuery& query) {
    std::vector<ByteStream> streams;
    streams.reserve(static_cast<std::size_t>(params.num_component_rings));
    for (int64_t k = 0; k < params.num_component_rings; ++k) {
        streams.push_back(pack_query_ring(params, query, k));
    }
    return streams;
}

SeededQuery unpack_query(const Params& params, const std::vector<ByteStream>& streams) {
    WireShape w = wire_shape(params);
    if (static_cast<int64_t>(streams.size()) != w.num_component_rings) {
        throw std::runtime_error("unpack_query: got " + std::to_string(streams.size()) + " streams, expected " +
                                 std::to_string(w.num_component_rings));
    }

    SeededQuery query;
    query.seeds.resize(static_cast<std::size_t>(w.num_component_rings));
    query.embedding_b_values.resize(static_cast<std::size_t>(w.num_component_rings));

    for (int64_t ring = 0; ring < w.num_component_rings; ++ring) {
        const auto k = static_cast<std::size_t>(ring);
        check_size(streams[k], query_stream_bytes(params, ring), "query");

        BitUnpacker in(streams[k]);
        in.get_bytes(query.seeds[k].data(), query.seeds[k].size());
        query.embedding_b_values[k] =
            read_coefficients(in, static_cast<std::size_t>(w.embedding_length), w.bits, params.q);

        if (ring == 0) {
            query.selector_b_values.reserve(static_cast<std::size_t>(w.num_clusters));
            for (int64_t c = 0; c < w.num_clusters; ++c) {
                query.selector_b_values.push_back(
                    read_coefficients(in, static_cast<std::size_t>(w.prime_digits), w.bits, params.q));
            }
        }
        in.expect_end();
    }
    return query;
}

// --- Response ---------------------------------------------------------------

ByteStream pack_response_ring(const Params& params, const std::vector<RLWECT>& ring_cts) {
    WireShape w = wire_shape(params);
    if (static_cast<int64_t>(ring_cts.size()) != w.splits_per_cluster) {
        throw std::invalid_argument("pack_response_ring: got " + std::to_string(ring_cts.size()) +
                                    " ciphertexts, expected splits_per_cluster = " +
                                    std::to_string(w.splits_per_cluster));
    }

    BitPacker out(response_stream_bytes(params));
    for (const auto& ct : ring_cts) {
        const Polynomial& a = ct.a();
        const Polynomial& b = ct.b();
        for (int64_t i = 0; i < w.n; ++i) {
            out.put(canonical(a[static_cast<std::size_t>(i)], params.q), w.bits);
        }
        for (int64_t i = 0; i < w.n; ++i) {
            out.put(canonical(b[static_cast<std::size_t>(i)], params.q), w.bits);
        }
    }
    return out.finish();
}

std::vector<RLWECT> unpack_response_ring(const CryptoContext& ctx, const Params& params, const ByteStream& stream) {
    check_size(stream, response_stream_bytes(params), "response");
    WireShape w = wire_shape(params);
    const auto n = static_cast<int32_t>(w.n);

    BitUnpacker in(stream);
    std::vector<RLWECT> cts;
    cts.reserve(static_cast<std::size_t>(w.splits_per_cluster));
    for (int64_t s = 0; s < w.splits_per_cluster; ++s) {
        Polynomial a(n, params.q);
        for (int32_t i = 0; i < n; ++i) {
            a[static_cast<std::size_t>(i)] = read_coefficient(in, w.bits, params.q);
        }
        Polynomial b(n, params.q);
        for (int32_t i = 0; i < n; ++i) {
            b[static_cast<std::size_t>(i)] = read_coefficient(in, w.bits, params.q);
        }
        cts.emplace_back(ctx.rlwe_param, std::move(a), std::move(b));
    }
    in.expect_end();
    return cts;
}

std::vector<ByteStream> pack_response(const Params& params, const std::vector<std::vector<RLWECT>>& cts) {
    if (static_cast<int64_t>(cts.size()) != params.num_component_rings) {
        throw std::invalid_argument("pack_response: expected one ciphertext vector per component ring");
    }
    std::vector<ByteStream> streams;
    streams.reserve(cts.size());
    for (const auto& ring_cts : cts) {
        streams.push_back(pack_response_ring(params, ring_cts));
    }
    return streams;
}

std::vector<std::vector<RLWECT>> unpack_response(const CryptoContext& ctx, const Params& params,
                                                 const std::vector<ByteStream>& streams) {
    if (static_cast<int64_t>(streams.size()) != params.num_component_rings) {
        throw std::runtime_error("unpack_response: got " + std::to_string(streams.size()) +
                                 " streams, expected " + std::to_string(params.num_component_rings));
    }
    std::vector<std::vector<RLWECT>> cts;
    cts.reserve(streams.size());
    for (const auto& stream : streams) {
        cts.push_back(unpack_response_ring(ctx, params, stream));
    }
    return cts;
}

} // namespace psearch
