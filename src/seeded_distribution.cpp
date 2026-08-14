#include "seeded_distribution.hpp"

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <cstring>
#include <stdexcept>

namespace psearch {

SeededUniformDistribution::SeededUniformDistribution(const std::array<uint8_t, kSeedBytes>& seed, int64_t from,
                                                       int64_t to)
    : from_(from), to_(to), buffer_(kRefillChunkBytes), buffer_pos_(kRefillChunkBytes) {
    if (to_ < from_) {
        throw std::invalid_argument("SeededUniformDistribution: to must be >= from");
    }
    // +1 for inclusive range; if from=0 and to=UINT64_MAX-ish this would
    // overflow, but our actual ranges (0..modulus, modulus < 2^63) are far
    // from that edge case.
    range_ = static_cast<uint64_t>(to_ - from_) + 1;

    // Zero IV/counter: fine here because the KEY (the seed) is fresh and
    // never reused (see generate_fresh_seed's contract) -- AES-CTR security
    // only requires the (key, counter) PAIR never repeat, and a fresh key
    // per seed with counter always starting at zero satisfies that as long
    // as seeds themselves are never reused.
    static const unsigned char zero_iv[16] = {0};

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    if (!ctx) {
        throw std::runtime_error("SeededUniformDistribution: EVP_CIPHER_CTX_new failed");
    }
    if (EVP_EncryptInit_ex(ctx, EVP_aes_128_ctr(), nullptr, seed.data(), zero_iv) != 1) {
        EVP_CIPHER_CTX_free(ctx);
        throw std::runtime_error("SeededUniformDistribution: EVP_EncryptInit_ex failed");
    }
    cipher_ctx_ = ctx;
}

SeededUniformDistribution::~SeededUniformDistribution() {
    if (cipher_ctx_) {
        EVP_CIPHER_CTX_free(static_cast<EVP_CIPHER_CTX*>(cipher_ctx_));
    }
}

void SeededUniformDistribution::refill_buffer() {
    // AES-CTR encryption of an all-zero plaintext buffer IS the keystream --
    // this is the standard trick for using a block cipher in CTR mode as a
    // deterministic CSPRNG.
    static thread_local std::vector<uint8_t> zeros(kRefillChunkBytes, 0);
    int out_len = 0;
    EVP_CIPHER_CTX* ctx = static_cast<EVP_CIPHER_CTX*>(cipher_ctx_);
    if (EVP_EncryptUpdate(ctx, buffer_.data(), &out_len, zeros.data(), static_cast<int>(kRefillChunkBytes)) != 1) {
        throw std::runtime_error("SeededUniformDistribution: EVP_EncryptUpdate failed");
    }
    if (static_cast<size_t>(out_len) != kRefillChunkBytes) {
        throw std::runtime_error("SeededUniformDistribution: unexpected keystream chunk size");
    }
    buffer_pos_ = 0;
}

uint64_t SeededUniformDistribution::next_raw_u64() {
    if (buffer_pos_ + 8 > buffer_.size()) {
        refill_buffer();
    }
    uint64_t value;
    std::memcpy(&value, buffer_.data() + buffer_pos_, 8);
    buffer_pos_ += 8;
    return value;
}

int64_t SeededUniformDistribution::next() {
    // Rejection sampling: reject values in the "overhang" beyond the
    // largest multiple of range_ that fits in 64 bits, so every output in
    // [0, range_) is equally likely.
    uint64_t limit = UINT64_MAX - (UINT64_MAX % range_);
    uint64_t raw;
    do {
        raw = next_raw_u64();
    } while (raw >= limit);
    return from_ + static_cast<int64_t>(raw % range_);
}

std::array<uint8_t, SeededUniformDistribution::kSeedBytes> generate_fresh_seed() {
    std::array<uint8_t, SeededUniformDistribution::kSeedBytes> seed;
    if (RAND_bytes(seed.data(), static_cast<int>(seed.size())) != 1) {
        throw std::runtime_error("generate_fresh_seed: RAND_bytes failed");
    }
    return seed;
}

FHEDeck::LWECT reconstruct_lwe_from_seed_and_b(std::shared_ptr<const FHEDeck::LWEParam> lwe_param,
                                                SeededUniformDistribution& a_stream, int64_t b_value) {
    FHEDeck::LWECT ct(lwe_param);
    ct[0] = b_value;
    for (int32_t i = 1; i <= lwe_param->dim(); ++i) {
        ct[i] = a_stream.next();
    }
    return ct;
}

} // namespace psearch