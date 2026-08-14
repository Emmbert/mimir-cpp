#include "seeded_eval_keys.hpp"

using namespace FHEDeck;

namespace psearch {

namespace {

/// Fills a Polynomial's coefficients by pulling `n` values from a
/// continuing seeded stream. Used for the `a` side, which is never sent --
/// only ever reconstructed this way, in the same order it was consumed
/// client-side.
Polynomial reconstruct_a_polynomial(SeededUniformDistribution& stream, int64_t n, int64_t q) {
    Polynomial a(n, q);
    for (int64_t j = 0; j < n; ++j) {
        a[j] = stream.next();
    }
    return a;
}

/// Builds a Polynomial directly from received coefficient data (the `b`
/// side, sent explicitly).
Polynomial polynomial_from_coefficients(const std::vector<int64_t>& coeffs, int64_t n, int64_t q) {
    Polynomial b(n, q);
    for (int64_t j = 0; j < n; ++j) {
        b[j] = coeffs[static_cast<size_t>(j)];
    }
    return b;
}

} // namespace

SeededClientPublicMaterial build_seeded_public_material(const CryptoContext& ctx, ClientSecretMaterial& secret) {
    auto eval_key_seed = generate_fresh_seed();
    auto eval_key_dist = std::make_shared<SeededUniformDistribution>(eval_key_seed, 0, ctx.rlwe_param->modulus());
    std::shared_ptr<Distribution> original_dist = secret.rlwe_sk->set_unif_dist(eval_key_dist);

    // Builds lwe_to_rlwe_ksk (automorphism keys) THEN lwe_to_rgsw_ksk
    // (message row, then message*sk row) -- exactly the order reconstruction
    // below has to replicate.
    ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

    secret.rlwe_sk->set_unif_dist(original_dist); // restore normal (non-deterministic) behaviour

    SeededClientPublicMaterial wire;
    wire.eval_key_seed = eval_key_seed;

    for (const auto& ext_ct_base : pub.lwe_to_rlwe_ksk->ext_key_content()) {
        // Only ExtendedRLWECT exists as a concrete implementation in this
        // codebase -- matches the static_cast<Concrete&>(base_ref) idiom
        // used pervasively elsewhere here (RLWECT, RLWEGadgetCT, etc.).
        const auto& ext_ct = static_cast<const ExtendedRLWECT&>(*ext_ct_base);
        wire.automorphism_b_values.push_back(ext_ct.get_b_coefficients());
    }

    const RLWEGadgetCT& ct_of_sk_dest = pub.lwe_to_rgsw_ksk->get_ct_of_sk_dest();
    wire.rgsw_message_row_b_values = ct_of_sk_dest.get_b_coefficients();
    wire.rgsw_message_sk_row_b_values = ct_of_sk_dest.get_b_sk_coefficients();

    return wire;
}

ClientPublicMaterial reconstruct_public_material(const CryptoContext& ctx, const Params& params,
                                                   const SeededClientPublicMaterial& wire) {
    SeededUniformDistribution a_stream(wire.eval_key_seed, 0, ctx.rlwe_param->modulus());

    // --- Automorphism keys: one ExtendedRLWECT per level. -------------------
    std::vector<std::shared_ptr<ExtendedPolynomialCT>> ext_key_content;
    ext_key_content.reserve(wire.automorphism_b_values.size());
    for (const auto& level_b_values : wire.automorphism_b_values) {
        std::vector<RLWECT> gadget_ct;
        gadget_ct.reserve(level_b_values.size());
        for (const auto& digit_b : level_b_values) {
            Polynomial a = reconstruct_a_polynomial(a_stream, params.n, params.q);
            Polynomial b = polynomial_from_coefficients(digit_b, params.n, params.q);
            gadget_ct.emplace_back(ctx.rlwe_param, a, b);
        }
        ext_key_content.push_back(std::make_shared<ExtendedRLWECT>(ctx.rlwe_param, ctx.gadget_ksk, gadget_ct));
    }
    auto lwe_to_rlwe_ksk = std::make_shared<LWEToRLWEKeySwitchKey>(ctx.rlwe_param, std::move(ext_key_content));

    // --- RGSW switch key's ct_of_sk_dest: message row, then message*sk row. -
    std::vector<RLWECT> message_row;
    message_row.reserve(wire.rgsw_message_row_b_values.size());
    for (const auto& digit_b : wire.rgsw_message_row_b_values) {
        Polynomial a = reconstruct_a_polynomial(a_stream, params.n, params.q);
        Polynomial b = polynomial_from_coefficients(digit_b, params.n, params.q);
        message_row.emplace_back(ctx.rlwe_param, a, b);
    }

    std::vector<RLWECT> message_sk_row;
    message_sk_row.reserve(wire.rgsw_message_sk_row_b_values.size());
    for (const auto& digit_b : wire.rgsw_message_sk_row_b_values) {
        Polynomial a = reconstruct_a_polynomial(a_stream, params.n, params.q);
        Polynomial b = polynomial_from_coefficients(digit_b, params.n, params.q);
        message_sk_row.emplace_back(ctx.rlwe_param, a, b);
    }

    RLWEGadgetCT ct_of_sk_dest(ctx.rlwe_param, ctx.gadget_rgsw, message_row, message_sk_row);

    auto lwe_to_rgsw_ksk = std::make_shared<LWEToRGSWKeySwitchKey>(
        lwe_to_rlwe_ksk, std::move(ct_of_sk_dest), ctx.rlwe_param, ctx.gadget_rgsw);

    return ClientPublicMaterial{lwe_to_rlwe_ksk, lwe_to_rgsw_ksk};
}

} // namespace psearch
