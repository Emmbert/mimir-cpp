#pragma once
#include <array>
#include <cstdint>
#include <vector>

#include "ciphertexts/lwe.h" // FHEDeck::LWECT, FHEDeck::LWEParam
#include "common/sample.h"   // FHEDeck::Distribution

namespace psearch {

/// A Distribution backed by a single continuous AES-128-CTR keystream,
/// keyed by a 128-bit seed. Drop-in replacement for FHEDeck's
/// StandardUniformIntegerDistribution: same next() interface (returns a
/// value uniformly distributed in [from, to] inclusive, matching how
/// LWESK/RLWESK construct their own m_unif_dist), but fully deterministic --
/// two SeededUniformDistribution objects built from the same seed and range
/// produce the identical sequence of values.
///
/// This is what gets swapped into LWESK::set_unif_dist / RLWESK::set_unif_dist
/// before building a batch of ciphertexts (one query, or one registration's
/// eval keys) that should be reconstructible from the seed alone. Since it's
/// one stateful object reused across every encrypt() call in that batch, the
/// underlying AES-CTR counter advances continuously across the whole batch --
/// no re-keying per ciphertext, which is what keeps this fast (see the
/// throughput estimate from the design discussion: ~3 GB/s on AES-NI
/// hardware, dominated by this property).
///
/// Uses rejection sampling to avoid modulo bias when mapping a raw 64-bit
/// keystream word into [from, to] -- for the ranges actually used here
/// (from=0, to=modulus, modulus close to 2^48-2^56), the bias from a naive
/// modulo would already be astronomically small, but rejection sampling
/// costs essentially nothing extra given the throughput headroom and avoids
/// having to reason about how small is "small enough."
class SeededUniformDistribution : public FHEDeck::Distribution {
public:
    static constexpr size_t kSeedBytes = 16; // 128 bits

    /// `seed` is the AES-128 key. `from`/`to` match
    /// StandardUniformIntegerDistribution's semantics exactly (inclusive on
    /// both ends).
    SeededUniformDistribution(const std::array<uint8_t, kSeedBytes>& seed, int64_t from, int64_t to);
    ~SeededUniformDistribution();

    // Not copyable (holds an OpenSSL cipher context) -- move if you need to
    // relocate one, or just construct in place / hold via shared_ptr.
    SeededUniformDistribution(const SeededUniformDistribution&) = delete;
    SeededUniformDistribution& operator=(const SeededUniformDistribution&) = delete;

    int64_t next() override;

private:
    int64_t from_;
    int64_t to_;
    uint64_t range_; // to_ - from_ + 1, as unsigned to avoid overflow at the top of the range

    void* cipher_ctx_; // EVP_CIPHER_CTX*, opaque here to keep OpenSSL out of this header

    // Small buffer of already-generated keystream bytes, refilled from the
    // AES-CTR stream in chunks rather than one EVP_EncryptUpdate call per
    // 8 bytes needed.
    std::vector<uint8_t> buffer_;
    size_t buffer_pos_;

    static constexpr size_t kRefillChunkBytes = 4096;

    void refill_buffer();
    uint64_t next_raw_u64();
};

/// Generates a fresh, cryptographically random 128-bit seed (from the OS's
/// CSPRNG, e.g. /dev/urandom via OpenSSL's RAND_bytes) -- use this once per
/// query and once per registration, NEVER reuse a seed. See the earlier
/// discussion: reusing "a" across two encryptions under the same key is a
/// textbook LWE break, structurally equivalent to two-time-pad reuse.
std::array<uint8_t, SeededUniformDistribution::kSeedBytes> generate_fresh_seed();

/// Rebuilds one LWE ciphertext from a continuing seeded stream (for `a`)
/// and a directly-received `b` value. `a_stream` must be shared across
/// every ciphertext being reconstructed in one batch, consumed in the SAME
/// order the ciphertexts were originally encrypted in -- constructing a
/// fresh SeededUniformDistribution per ciphertext would restart the
/// keystream each time, producing identical (wrong) `a` values, and is
/// exactly the kind of randomness reuse that breaks LWE's security.
///
/// This is the ONE canonical implementation of this reconstruction --
/// shared by seeded_query.cpp's reconstruct_query and every test that needs
/// to reconstruct a raw LWE ciphertext directly, so a passing test is
/// evidence about the exact code path production/benchmark code runs, not a
/// structurally-similar reimplementation that could silently drift.
FHEDeck::LWECT reconstruct_lwe_from_seed_and_b(std::shared_ptr<const FHEDeck::LWEParam> lwe_param,
                                                SeededUniformDistribution& a_stream, int64_t b_value);

} // namespace psearch