#include "wire_protocol.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

using namespace FHEDeck;

namespace psearch {

namespace {

// --- Low-level byte-buffer writer/reader helpers. --------------------------

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + sizeof(v));
}

void write_i64(std::vector<uint8_t>& buf, int64_t v) {
    buf.insert(buf.end(), reinterpret_cast<const uint8_t*>(&v), reinterpret_cast<const uint8_t*>(&v) + sizeof(v));
}

void write_bytes(std::vector<uint8_t>& buf, const uint8_t* data, size_t len) {
    buf.insert(buf.end(), data, data + len);
}

/// [count][values...]
void write_vec_i64(std::vector<uint8_t>& buf, const std::vector<int64_t>& vec) {
    write_u32(buf, static_cast<uint32_t>(vec.size()));
    for (int64_t v : vec) {
        write_i64(buf, v);
    }
}

/// [count][write_vec_i64 for each]
void write_vec_vec_i64(std::vector<uint8_t>& buf, const std::vector<std::vector<int64_t>>& vec) {
    write_u32(buf, static_cast<uint32_t>(vec.size()));
    for (const auto& inner : vec) {
        write_vec_i64(buf, inner);
    }
}

/// [count][write_vec_vec_i64 for each]
void write_vec_vec_vec_i64(std::vector<uint8_t>& buf, const std::vector<std::vector<std::vector<int64_t>>>& vec) {
    write_u32(buf, static_cast<uint32_t>(vec.size()));
    for (const auto& inner : vec) {
        write_vec_vec_i64(buf, inner);
    }
}

/// Cursor-based reader over an existing buffer. Throws on underrun rather
/// than reading past the end -- a malformed/truncated message should fail
/// loudly, not read garbage or crash.
class ByteReader {
public:
    explicit ByteReader(const std::vector<uint8_t>& buf) : buf_(buf), pos_(0) {}

    uint32_t read_u32() {
        require(sizeof(uint32_t));
        uint32_t v;
        std::memcpy(&v, buf_.data() + pos_, sizeof(v));
        pos_ += sizeof(v);
        return v;
    }

    int64_t read_i64() {
        require(sizeof(int64_t));
        int64_t v;
        std::memcpy(&v, buf_.data() + pos_, sizeof(v));
        pos_ += sizeof(v);
        return v;
    }

    void read_bytes(uint8_t* out, size_t len) {
        require(len);
        std::memcpy(out, buf_.data() + pos_, len);
        pos_ += len;
    }

    std::vector<int64_t> read_vec_i64() {
        uint32_t count = read_u32();
        std::vector<int64_t> out;
        out.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(read_i64());
        }
        return out;
    }

    std::vector<std::vector<int64_t>> read_vec_vec_i64() {
        uint32_t count = read_u32();
        std::vector<std::vector<int64_t>> out;
        out.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(read_vec_i64());
        }
        return out;
    }

    std::vector<std::vector<std::vector<int64_t>>> read_vec_vec_vec_i64() {
        uint32_t count = read_u32();
        std::vector<std::vector<std::vector<int64_t>>> out;
        out.reserve(count);
        for (uint32_t i = 0; i < count; ++i) {
            out.push_back(read_vec_vec_i64());
        }
        return out;
    }

private:
    void require(size_t len) const {
        if (pos_ + len > buf_.size()) {
            throw std::runtime_error("ByteReader: buffer underrun -- message truncated or malformed");
        }
    }

    const std::vector<uint8_t>& buf_;
    size_t pos_;
};

} // namespace

std::vector<uint8_t> serialize_seeded_query(const SeededQuery& query) {
    std::vector<uint8_t> buf;
    write_bytes(buf, query.seed.data(), query.seed.size());
    write_vec_i64(buf, query.embedding_b_values);
    write_vec_vec_i64(buf, query.selector_b_values);
    return buf;
}

SeededQuery deserialize_seeded_query(const std::vector<uint8_t>& buf) {
    ByteReader r(buf);
    SeededQuery query;
    r.read_bytes(query.seed.data(), query.seed.size());
    query.embedding_b_values = r.read_vec_i64();
    query.selector_b_values = r.read_vec_vec_i64();
    return query;
}

std::vector<uint8_t> serialize_seeded_public_material(const SeededClientPublicMaterial& material) {
    std::vector<uint8_t> buf;
    write_bytes(buf, material.eval_key_seed.data(), material.eval_key_seed.size());
    write_vec_vec_vec_i64(buf, material.automorphism_b_values);
    write_vec_vec_i64(buf, material.rgsw_message_row_b_values);
    write_vec_vec_i64(buf, material.rgsw_message_sk_row_b_values);
    return buf;
}

SeededClientPublicMaterial deserialize_seeded_public_material(const std::vector<uint8_t>& buf) {
    ByteReader r(buf);
    SeededClientPublicMaterial material;
    r.read_bytes(material.eval_key_seed.data(), material.eval_key_seed.size());
    material.automorphism_b_values = r.read_vec_vec_vec_i64();
    material.rgsw_message_row_b_values = r.read_vec_vec_i64();
    material.rgsw_message_sk_row_b_values = r.read_vec_vec_i64();
    return material;
}

std::vector<uint8_t> serialize_query_response(const CryptoContext& ctx, const QueryResponse& response) {
    std::vector<uint8_t> buf;
    write_u32(buf, static_cast<uint32_t>(response.ciphertexts.size()));
    int64_t n = static_cast<int64_t>(ctx.rlwe_param->size());
    for (const auto& ct : response.ciphertexts) {
        for (int64_t i = 0; i < n; ++i) {
            write_i64(buf, ct.a()[i]);
        }
        for (int64_t i = 0; i < n; ++i) {
            write_i64(buf, ct.b()[i]);
        }
    }
    return buf;
}

QueryResponse deserialize_query_response(const CryptoContext& ctx, const std::vector<uint8_t>& buf) {
    ByteReader r(buf);
    uint32_t count = r.read_u32();
    int64_t n = static_cast<int64_t>(ctx.rlwe_param->size());
    uint64_t q = ctx.rlwe_param->modulus();

    QueryResponse response;
    response.ciphertexts.reserve(count);
    for (uint32_t k = 0; k < count; ++k) {
        Polynomial a(n, q);
        for (int64_t i = 0; i < n; ++i) {
            a[i] = r.read_i64();
        }
        Polynomial b(n, q);
        for (int64_t i = 0; i < n; ++i) {
            b[i] = r.read_i64();
        }
        response.ciphertexts.emplace_back(ctx.rlwe_param, a, b);
    }
    return response;
}

std::vector<uint8_t> serialize_registration_message(const RegistrationMessage& msg) {
    std::vector<uint8_t> buf;
    write_bytes(buf, msg.session_id.data(), msg.session_id.size());
    std::vector<uint8_t> material_bytes = serialize_seeded_public_material(msg.material);
    buf.insert(buf.end(), material_bytes.begin(), material_bytes.end());
    return buf;
}

RegistrationMessage deserialize_registration_message(const std::vector<uint8_t>& buf) {
    if (buf.size() < kSessionIdBytes) {
        throw std::runtime_error("deserialize_registration_message: buffer too short for a session ID");
    }
    RegistrationMessage msg;
    std::copy(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(kSessionIdBytes), msg.session_id.begin());

    std::vector<uint8_t> material_bytes(buf.begin() + static_cast<std::ptrdiff_t>(kSessionIdBytes), buf.end());
    msg.material = deserialize_seeded_public_material(material_bytes);
    return msg;
}

std::vector<uint8_t> serialize_query_message(const QueryMessage& msg) {
    std::vector<uint8_t> buf;
    write_bytes(buf, msg.session_id.data(), msg.session_id.size());
    std::vector<uint8_t> query_bytes = serialize_seeded_query(msg.query);
    buf.insert(buf.end(), query_bytes.begin(), query_bytes.end());
    return buf;
}

QueryMessage deserialize_query_message(const std::vector<uint8_t>& buf) {
    if (buf.size() < kSessionIdBytes) {
        throw std::runtime_error("deserialize_query_message: buffer too short for a session ID");
    }
    QueryMessage msg;
    std::copy(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(kSessionIdBytes), msg.session_id.begin());

    std::vector<uint8_t> query_bytes(buf.begin() + static_cast<std::ptrdiff_t>(kSessionIdBytes), buf.end());
    msg.query = deserialize_seeded_query(query_bytes);
    return msg;
}

} // namespace psearch
