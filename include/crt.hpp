#pragma once
#include <cstdint>
#include <utility>

namespace psearch {

/// CRT decomposition for the specific two-component-ring construction this
/// project uses: p1 = comp_ring_modulus, p2 = comp_ring_modulus - 1.
/// Consecutive integers are always coprime, so this pair is always a valid
/// CRT basis without needing a primality check on either component.
///
/// This specific choice (p2 = p1 - 1, not an arbitrary coprime pair) is what
/// makes reconstruction cheap: p1 mod p2 is ALWAYS exactly 1 (since
/// p1 = 1*p2 + 1 by construction), so the modular inverse Garner's formula
/// normally needs is trivially 1 -- no extended Euclidean algorithm, just a
/// subtract/mod/multiply/add. If this project ever needs a general
/// coprime-but-not-consecutive pair, this fast path would need generalizing
/// back to a real modular inverse; it is NOT correct for an arbitrary
/// coprime (p1, p2).

/// Splits a value v in [0, p1*(p1-1)) into its CRT components
/// (v mod p1, v mod (p1-1)).
inline std::pair<int64_t, int64_t> crt_split(int64_t v, int64_t comp_ring_modulus) {
    int64_t p1 = comp_ring_modulus;
    int64_t p2 = comp_ring_modulus - 1;
    return {v % p1, v % p2};
}

/// Reconstructs v in [0, p1*(p1-1)) from its CRT components (r1 mod p1,
/// r2 mod p2), where p2 = p1 - 1. Fast path exploiting p1 mod p2 == 1 --
/// see the file-level comment above. r1 must be in [0, p1), r2 in [0, p2).
inline int64_t crt_recompose(int64_t r1, int64_t r2, int64_t comp_ring_modulus) {
    int64_t p1 = comp_ring_modulus;
    int64_t p2 = comp_ring_modulus - 1;
    int64_t diff = (r2 - r1) % p2;
    if (diff < 0) {
        diff += p2;
    }
    return r1 + p1 * diff;
}

/// The combined modulus p1*(p1-1) -- the actual message-space size once
/// both CRT components are considered together. This is what must be
/// compared against PLAINTEXT_MODULUS (the lower bound from the parameter
/// file) when validating CRT parameters at load time.
inline int64_t crt_combined_modulus(int64_t comp_ring_modulus) {
    return comp_ring_modulus * (comp_ring_modulus - 1);
}

} // namespace psearch
