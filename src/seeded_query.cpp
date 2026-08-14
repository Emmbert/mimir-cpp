#include "seeded_query.hpp"

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

SeededQuery build_seeded_query(const CryptoContext& ctx, ClientSecretMaterial& secret,
                                const std::vector<int64_t>& embedding_values, int64_t num_clusters,
                                int64_t desired_cluster_index) {
    auto seed = generate_fresh_seed();
    auto dist = std::make_shared<SeededUniformDistribution>(seed, 0, ctx.rlwe_param->modulus());
    std::shared_ptr<Distribution> original_dist = secret.lwe_sk->set_unif_dist(dist);

    SeededQuery wire;
    wire.seed = seed;

    // Embedding ciphertexts, in order -- consumed from the stream first.
    wire.embedding_b_values.reserve(embedding_values.size());
    for (int64_t m : embedding_values) {
        LWECT ct = secret.lwe_sk->encode_and_encrypt(m, ctx.encoding);
        wire.embedding_b_values.push_back(ct[0]);
    }

    // Selector unit vector, cluster by cluster -- consumed second. Uses
    // secret.lwe_gadget_sk, which wraps the SAME secret.lwe_sk object, so
    // this continues the identical stream rather than starting a new one.
    wire.selector_b_values.reserve(static_cast<size_t>(num_clusters));
    for (int64_t c = 0; c < num_clusters; ++c) {
        int64_t bit = (c == desired_cluster_index) ? 1 : 0;
        LWEGadgetCT gadget_ct = secret.lwe_gadget_sk->gadget_encrypt(bit);
        wire.selector_b_values.push_back(gadget_ct.get_b_values());
    }

    secret.lwe_sk->set_unif_dist(original_dist); // restore normal (non-deterministic) behaviour
    return wire;
}

ReconstructedQuery reconstruct_query(const CryptoContext& ctx, const Params& params, const SeededQuery& wire) {
    auto lwe_param = lwe_param_from_ctx(ctx);
    SeededUniformDistribution a_stream(wire.seed, 0, ctx.rlwe_param->modulus());

    ReconstructedQuery result;

    result.embedding_cts.reserve(wire.embedding_b_values.size());
    for (int64_t b : wire.embedding_b_values) {
        result.embedding_cts.push_back(reconstruct_lwe_from_seed_and_b(lwe_param, a_stream, b));
    }

    result.selector_cts.reserve(wire.selector_b_values.size());
    for (const auto& digit_b_values : wire.selector_b_values) {
        std::vector<LWECT> ct_content;
        ct_content.reserve(digit_b_values.size());
        for (int64_t b : digit_b_values) {
            ct_content.push_back(reconstruct_lwe_from_seed_and_b(lwe_param, a_stream, b));
        }
        result.selector_cts.emplace_back(lwe_param, params.decomposition_base_prime, std::move(ct_content));
    }

    return result;
}

} // namespace psearch