#include "seeded_query.hpp"

#include <omp.h>
#include <stdexcept>

#include "crt.hpp"

using namespace FHEDeck;

namespace psearch {

namespace {

/// Fresh LWEParam matching the client's LWE secret key's own -- built from
/// the (public) ring parameters alone, no secret key needed. Mirrors
/// RLWESK::extract_lwe_key's own construction (dim = n, modulus = q).
std::shared_ptr<const LWEParam> lwe_param_from_ctx(const CryptoContext& ctx) {
    return std::make_shared<const LWEParam>(static_cast<int32_t>(ctx.rlwe_param->size()), ctx.rlwe_param->modulus());
}

} // namespace

SeededQuery build_seeded_query(const CryptoContext& ctx, const Params& params, ClientSecretMaterial& secret,
                                const std::vector<int64_t>& embedding_values, int64_t desired_cluster_index) {
    int64_t r = params.num_component_rings;
    if (static_cast<int64_t>(ctx.component_encodings.size()) != r) {
        throw std::invalid_argument(
            "build_seeded_query: ctx.component_encodings.size() (" +
            std::to_string(ctx.component_encodings.size()) + ") must equal params.num_component_rings (" +
            std::to_string(r) + ") -- was ctx built via CryptoContext::from_params(params) with this SAME params?");
    }

    SeededQuery wire;
    wire.seeds.resize(static_cast<size_t>(r));
    wire.embedding_b_values.resize(static_cast<size_t>(r));

    std::vector<std::shared_ptr<SeededUniformDistribution>> dists(static_cast<size_t>(r));
    for (int64_t k = 0; k < r; ++k) {
        wire.seeds[static_cast<size_t>(k)] = generate_fresh_seed();
        dists[static_cast<size_t>(k)] =
            std::make_shared<SeededUniformDistribution>(wire.seeds[static_cast<size_t>(k)], 0, ctx.rlwe_param->modulus());
        wire.embedding_b_values[static_cast<size_t>(k)].reserve(embedding_values.size());
    }

    std::shared_ptr<Distribution> original_dist = secret.lwe_sk->set_unif_dist(dists[0]); // placeholder; reset per component below

    // --- Embedding ciphertexts: CRT-split (if r==2) then encrypt each
    // component on its own stream. NOT parallelized across k, unlike
    // reconstruct_query's equivalent loop: secret.lwe_sk->set_unif_dist()
    // below mutates ONE shared object's internal state, so two threads
    // calling it concurrently for different k would race, potentially
    // encrypting a component under the WRONG stream's randomness silently
    // (no crash, just quiet corruption -- see the discussion this was
    // written alongside). Making this safe needs each stream to have its
    // own independent encryption capability instead of sharing one mutable
    // secret.lwe_sk -- deliberately deferred, not implemented here. -----------
    for (int64_t j = 0; j < static_cast<int64_t>(embedding_values.size()); ++j) {
        int64_t v = embedding_values[static_cast<size_t>(j)];

        std::vector<int64_t> components;
        if (r == 1) {
            components = {v};
        } else { // r == 2
            auto [c1, c2] = crt_split(v, params.comp_ring_modulus);
            components = {c1, c2};
        }

        for (int64_t k = 0; k < r; ++k) {
            secret.lwe_sk->set_unif_dist(dists[static_cast<size_t>(k)]);
            LWECT ct = secret.lwe_sk->encode_and_encrypt(components[static_cast<size_t>(k)],
                                                          ctx.component_encodings[static_cast<size_t>(k)]);
            wire.embedding_b_values[static_cast<size_t>(k)].push_back(ct[0]);
        }
    }

    // --- Selector unit vector: continues seeds[0]'s stream. secret.lwe_gadget_sk
    // wraps the SAME secret.lwe_sk object, so setting the distribution here
    // covers both. -----------------------------------------------------------------
    secret.lwe_sk->set_unif_dist(dists[0]);
    wire.selector_b_values.reserve(static_cast<size_t>(params.num_clusters));
    for (int64_t c = 0; c < params.num_clusters; ++c) {
        int64_t bit = (c == desired_cluster_index) ? 1 : 0;
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        wire.selector_b_values.push_back(gadget_ct.get_b_values());
    }

    secret.lwe_sk->set_unif_dist(original_dist); // restore normal (non-deterministic) behaviour
    return wire;
}

ReconstructedQuery reconstruct_query(const CryptoContext& ctx, const Params& params, const SeededQuery& wire) {
    int64_t r = params.num_component_rings;
    if (static_cast<int64_t>(wire.seeds.size()) != r || static_cast<int64_t>(wire.embedding_b_values.size()) != r) {
        throw std::invalid_argument("reconstruct_query: wire.seeds/embedding_b_values size does not match "
                                     "params.num_component_rings (" +
                                     std::to_string(r) + ")");
    }

    auto lwe_param = lwe_param_from_ctx(ctx);

    std::vector<std::shared_ptr<SeededUniformDistribution>> dists(static_cast<size_t>(r));
    for (int64_t k = 0; k < r; ++k) {
        dists[static_cast<size_t>(k)] =
            std::make_shared<SeededUniformDistribution>(wire.seeds[static_cast<size_t>(k)], 0, ctx.rlwe_param->modulus());
    }

    ReconstructedQuery result;
    result.embedding_cts.resize(static_cast<size_t>(r));

    // Parallel across component rings -- safe because each iteration only
    // touches its own independent dists[k] object and writes to its own
    // result.embedding_cts[k]; no shared mutable state between iterations
    // (unlike build_seeded_query's client-side loop, which mutates a single
    // shared secret.lwe_sk's distribution in place and is NOT currently
    // safe to parallelize the same way -- see seeded_query.hpp). At most 2
    // iterations (r is always 1 or 2), so this is a small win, not a large
    // one -- but it's genuinely safe, unlike the client-side case.
    #pragma omp parallel for if(r > 1)
    for (int64_t k = 0; k < r; ++k) {
        result.embedding_cts[static_cast<size_t>(k)].reserve(wire.embedding_b_values[static_cast<size_t>(k)].size());
        for (int64_t b : wire.embedding_b_values[static_cast<size_t>(k)]) {
            result.embedding_cts[static_cast<size_t>(k)].push_back(
                reconstruct_lwe_from_seed_and_b(lwe_param, *dists[static_cast<size_t>(k)], b));
        }
    }

    // --- Selector: continues seeds[0]'s stream (dists[0]), AFTER its
    // embeddings have already been consumed above -- matches
    // build_seeded_query's exact order. ------------------------------------------
    result.selector_cts.reserve(wire.selector_b_values.size());
    for (const auto& digit_b_values : wire.selector_b_values) {
        std::vector<LWECT> ct_content;
        ct_content.reserve(digit_b_values.size());
        for (int64_t b : digit_b_values) {
            ct_content.push_back(reconstruct_lwe_from_seed_and_b(lwe_param, *dists[0], b));
        }
        result.selector_cts.emplace_back(lwe_param, params.decomposition_base_prime, std::move(ct_content));
    }

    return result;
}

} // namespace psearch