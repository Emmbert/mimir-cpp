#pragma once
#include <array>
#include <vector>

#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"

namespace psearch {

/// The seed-compressed form of one query: one seed covering a continuous
/// stream, plus only the b-values that were computed from it -- everything
/// a client actually needs to send. Embedding ciphertexts are consumed from
/// the stream first, then the selector ciphertexts cluster-by-cluster --
/// matching build_seeded_query's own generation order exactly, since
/// secret.lwe_gadget_sk wraps the SAME secret.lwe_sk object, so swapping
/// one distribution covers both.
struct SeededQuery {
    std::array<uint8_t, SeededUniformDistribution::kSeedBytes> seed;
    std::vector<int64_t> embedding_b_values;              // length l
    std::vector<std::vector<int64_t>> selector_b_values;  // [cluster][digit]
};

/// What reconstruct_query produces: real, usable ciphertexts, ready to be
/// switched via a (also reconstructed, or original) ClientPublicMaterial.
struct ReconstructedQuery {
    std::vector<FHEDeck::LWECT> embedding_cts;
    std::vector<FHEDeck::LWEGadgetCT> selector_cts; // one per cluster
};

/// Client-side: encrypts `embedding_values` (length l) and a selector unit
/// vector (num_clusters entries, all 0 except a 1 at desired_cluster_index)
/// using a fresh seed, swapping secret.lwe_sk's distribution for the
/// duration and restoring it afterward. Returns ONLY the wire-compressed
/// form -- the real ciphertexts built along the way are not kept.
SeededQuery build_seeded_query(const CryptoContext& ctx, ClientSecretMaterial& secret,
                                const std::vector<int64_t>& embedding_values, int64_t num_clusters,
                                int64_t desired_cluster_index);

/// Server-side: reconstructs real, usable ciphertexts from the wire data
/// alone -- no secret key involved anywhere in this function.
ReconstructedQuery reconstruct_query(const CryptoContext& ctx, const Params& params, const SeededQuery& wire);

} // namespace psearch
