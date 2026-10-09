#pragma once
#include <array>
#include <vector>

#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"

namespace psearch {

/// The seed-compressed form of a client's eval keys (registration-time),
/// mirroring the layout of the actual objects: automorphism keys
/// (LWEToRLWEKeySwitchKey's ext_key_content) and the scheme switching key
/// RLWE'(sk^2) used by LWEToRGSWKeySwitchKey (ePrint 2023/112, Sec. 3.1).
/// Only ONE seed covers the
/// whole batch -- both pieces are generated from the same continuing
/// SeededUniformDistribution, in the same order generate_client_public_material
/// already builds them in (automorphism keys first, then the RGSW switch
/// key). See build_seeded_public_material / reconstruct_public_material.
struct SeededClientPublicMaterial {
    std::array<uint8_t, SeededUniformDistribution::kSeedBytes> eval_key_seed;

    /// [automorphism_level][digit][coefficient]. automorphism_level in
    /// [0, log2(n)), matching LWEToRLWEKeySwitchKey::key_switching_key_gen's
    /// own loop order (i = 2, 4, 8, ..., n).
    std::vector<std::vector<std::vector<int64_t>>> automorphism_b_values;

    /// RLWE'(sk^2) = (RLWE(B'^i * sk^2))_i, [digit][coefficient], d' digits.
    /// Replaces the former RGSW(sk) = (RLWE'(sk), RLWE'(sk^2)), which needed
    /// 2 * d' b-polynomials.
    std::vector<std::vector<int64_t>> rgsw_sk_squared_b_values;
};

/// Client-side: builds real eval keys using a fresh seed (swapping
/// secret.rlwe_sk's distribution for the duration, then restoring it), and
/// extracts the seed-compressed wire representation from them. The full
/// LWEToRLWEKeySwitchKey/LWEToRGSWKeySwitchKey objects built along the way
/// are NOT returned -- only what would actually be sent.
SeededClientPublicMaterial build_seeded_public_material(const CryptoContext& ctx, ClientSecretMaterial& secret);

/// Server-side: reconstructs a fully usable ClientPublicMaterial from the
/// wire data alone -- no secret key involved anywhere in this function.
ClientPublicMaterial reconstruct_public_material(const CryptoContext& ctx, const Params& params,
                                                   const SeededClientPublicMaterial& wire);

} // namespace psearch
