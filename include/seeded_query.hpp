#pragma once
#include <array>
#include <vector>

#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "seeded_distribution.hpp"

namespace psearch {

/// The seed-compressed form of one query. ONE seed/stream per CRT component
/// ring (params.num_component_rings entries) -- each stream covers that
/// ring's embedding vector. seeds[0]'s stream ADDITIONALLY continues on to
/// cover the selector ciphertexts afterward (the selector is never CRT-split
/// -- always exactly one selector set regardless of num_component_rings, so
/// there's no reason for it to need its own stream; it simply continues
/// whichever stream already exists).
///
/// When num_component_rings == 1, this is structurally IDENTICAL to the
/// original single-seed design -- seeds.size() == 1, embedding_b_values has
/// exactly one inner vector -- nothing consuming this struct needs a
/// separate CRT/non-CRT code path; the num_component_rings==1 case falls
/// out as a length-1 loop everywhere.
///
/// Two independent streams (rather than one continuous stream covering both
/// rings) is deliberate: SeededUniformDistribution's rejection sampling
/// makes each ciphertext's byte offset in a stream depend on exactly how
/// much every prior ciphertext in that SAME stream consumed, so unpacking
/// (or building) a single stream is inherently sequential. Two independent
/// streams means ring 0's and ring 1's embedding vectors can eventually be
/// built/unpacked in parallel with each other (not yet implemented -- see
/// build_seeded_query/reconstruct_query). Reusing the SAME seed for both
/// rings (with ring 1 continuing further into ring 0's stream) is NOT a
/// security problem in itself -- rejection sampling always draws fresh
/// pseudorandom values regardless of stream position -- but it would make
/// the two rings' unpacking dependent on each other again, undoing the
/// parallelism this structure exists to enable.
struct SeededQuery {
    std::vector<std::array<uint8_t, SeededUniformDistribution::kSeedBytes>> seeds;
    std::vector<std::vector<int64_t>> embedding_b_values; // [component_ring][j]
    std::vector<std::vector<int64_t>> selector_b_values;  // [cluster][digit] -- continues seeds[0]
};

/// What reconstruct_query produces: real, usable ciphertexts, ready to be
/// switched via a (also reconstructed, or original) ClientPublicMaterial.
struct ReconstructedQuery {
    std::vector<std::vector<FHEDeck::LWECT>> embedding_cts; // [component_ring][j]
    std::vector<FHEDeck::LWEGadgetCT> selector_cts;         // one per cluster
};

/// Client-side: encrypts `embedding_values` (length l, values in
/// [0, params.combined_component_ring_modulus) if CRT, or the usual
/// plaintext_modulus range otherwise) and a selector unit vector
/// (params.num_clusters entries, all 0 except a 1 at desired_cluster_index).
/// desired_cluster_index is taken as an explicit parameter, NOT read from
/// params.desired_cluster_index, even though Params also has a
/// same-named field -- that field exists only as a testing/benchmarking
/// convenience for pinning one fixed scenario onto a Params object; a real
/// client computes desired_cluster_index fresh per query (nearest-centroid
/// lookup against whatever was just searched for), so it's a genuine
/// per-call input, not a deployment-shape constant the way num_clusters is.
/// num_clusters itself is NOT a separate parameter -- params.num_clusters
/// is used directly, since there's no scenario where a caller would
/// legitimately want a different value than what's actually in params.
///
/// Internally CRT-splits each embedding value when params.num_component_rings
/// == 2, encrypting each component under its own PlaintextEncoding
/// (ctx.component_encodings) and its own seeded stream. Swaps
/// secret.lwe_sk's distribution for the duration (once per stream) and
/// restores it afterward. Returns ONLY the wire-compressed form -- the real
/// ciphertexts built along the way are not kept.
SeededQuery build_seeded_query(const CryptoContext& ctx, const Params& params, ClientSecretMaterial& secret,
                                const std::vector<int64_t>& embedding_values, int64_t desired_cluster_index);

/// Server-side: reconstructs real, usable ciphertexts from the wire data
/// alone -- no secret key involved anywhere in this function.
ReconstructedQuery reconstruct_query(const CryptoContext& ctx, const Params& params, const SeededQuery& wire);

} // namespace psearch