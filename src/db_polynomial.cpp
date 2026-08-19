#include "db_polynomial.hpp"

#include <stdexcept>

namespace psearch {

int64_t min_embedding_value(const Params& params) {
    return -(int64_t(1) << (params.embedding_precision - 1));
}

int64_t max_embedding_value(const Params& params) {
    return (int64_t(1) << (params.embedding_precision - 1)) - 1;
}

int64_t max_abs_embedding_value(const Params& params) {
    return -min_embedding_value(params); // magnitude of the lower bound is always the larger one
}

bool products_can_overflow(const Params& params) {
    int64_t max_abs = max_abs_embedding_value(params);
    // Worst case: both factors at max magnitude, e.g. (-8)*(-8) = 64.
    // Threshold uses (modulus+1)/2, not modulus/2: for ODD modulus,
    // centered_residue's representable range is exactly symmetric,
    // [-floor(p/2), +floor(p/2)], so a worst case landing exactly at
    // floor(p/2) is still safe -- floor division alone would reject it one
    // too early. For EVEN modulus this is a no-op: (p+1)/2 == p/2 there,
    // since integer division floors (p+1) (odd) to the same value.
    return (max_abs * max_abs) >= ((params.plaintext_modulus + 1) / 2);
}

bool dot_product_can_overflow(const Params& params) {
    __int128 max_abs = max_abs_embedding_value(params);
    // Worst case: every one of embedding_length terms at max magnitude with
    // matching signs, e.g. l * (-8)*(-8). Threshold uses (modulus+1)/2, not
    // modulus/2 -- see products_can_overflow's comment above for why: this
    // is a no-op for even modulus, and fixes a genuine off-by-one for odd
    // modulus where the worst case can legitimately land exactly at
    // floor(p/2) and still be safe.
    __int128 worst_case = max_abs * max_abs * static_cast<__int128>(params.embedding_length);
    return worst_case >= static_cast<__int128>((params.plaintext_modulus + 1) / 2);
}

int64_t reduce_mod(int64_t value, int64_t modulus) {
    int64_t reduced = value % modulus;
    if (reduced < 0) {
        reduced += modulus;
    }
    return reduced;
}

int64_t centered_residue(int64_t value, int64_t modulus) {
    if (value > modulus / 2) {
        return value - modulus;
    }
    return value;
}

int64_t decode_to_signed(int64_t value, const Params& params) {
    return centered_residue(value, params.plaintext_modulus);
}

std::vector<int64_t> centered_residues(const FHEDeck::Vector& values, int64_t modulus) {
    std::vector<int64_t> out(static_cast<size_t>(values.size()));
    for (int64_t i = 0; i < values.size(); ++i) {
        out[static_cast<size_t>(i)] = centered_residue(values[i], modulus);
    }
    return out;
}

std::vector<int64_t> decode_to_signed(const FHEDeck::Vector& values, const Params& params) {
    return centered_residues(values, params.plaintext_modulus);
}

SignedValue sample_signed_value(const Params& params, std::mt19937_64& rng) {
    std::uniform_int_distribution<int64_t> dist(min_embedding_value(params), max_embedding_value(params));
    int64_t raw = dist(rng);
    //int64_t raw = -1; // for debugging
    //std::cout << "[log] sampling raw message " << raw << ")...\n" << std::flush;
    return SignedValue{raw, reduce_mod(raw, params.plaintext_modulus)};
}

int64_t sample_signed_mod_value(const Params& params, std::mt19937_64& rng) {
    return sample_signed_value(params, rng).reduced;
}

DatabasePolynomial build_random_database_polynomial(const Params& params, std::mt19937_64& rng) {
    // Modulus q (the RLWE ring's own modulus), matching the convention used
    // throughout FHE-Deck's own internal RLWE-ring polynomials.
    FHEDeck::Polynomial poly(params.n, params.q);
    std::vector<int64_t> raw_values(static_cast<size_t>(params.n));

    for (int64_t i = 0; i < params.n; ++i) {
        SignedValue v = sample_signed_value(params, rng);
        poly[i] = v.reduced;
        raw_values[static_cast<size_t>(i)] = v.raw;
    }
    return DatabasePolynomial{std::move(poly), std::move(raw_values)};
}

FHEDeck::Polynomial build_polynomial_from_raw_values(const Params& params, const std::vector<int64_t>& raw_values,
                                                       int64_t modulus) {
    if (static_cast<int64_t>(raw_values.size()) != params.n) {
        throw std::invalid_argument("build_polynomial_from_raw_values: raw_values.size() (" +
                                     std::to_string(raw_values.size()) + ") must equal params.n (" +
                                     std::to_string(params.n) + ")");
    }
    FHEDeck::Polynomial poly(params.n, params.q);
    for (int64_t i = 0; i < params.n; ++i) {
        poly[i] = reduce_mod(raw_values[static_cast<size_t>(i)], modulus);
    }
    return poly;
}

int64_t component_ring_modulus(const Params& params, int64_t ring) {
    if (params.num_component_rings == 1) {
        if (ring != 0) {
            throw std::invalid_argument("component_ring_modulus: ring must be 0 when num_component_rings == 1");
        }
        return params.plaintext_modulus;
    }
    if (ring == 0) return params.comp_ring_modulus;
    if (ring == 1) return params.comp_ring_modulus - 1;
    throw std::invalid_argument("component_ring_modulus: ring must be 0 or 1 when num_component_rings == 2");
}

std::vector<FHEDeck::Polynomial> crt_split_database_polynomial(const Params& params,
                                                                 const std::vector<int64_t>& raw_values) {
    int64_t r = params.num_component_rings;
    std::vector<FHEDeck::Polynomial> result;
    result.reserve(static_cast<size_t>(r));
    for (int64_t ring = 0; ring < r; ++ring) {
        result.push_back(build_polynomial_from_raw_values(params, raw_values, component_ring_modulus(params, ring)));
    }
    return result;
}

DatabasePolynomialEvalForm build_database_polynomial_eval_form_from_raw_values(const CryptoContext& ctx,
                                                                                const Params& params,
                                                                                const std::vector<int64_t>& raw_values,
                                                                                int64_t modulus) {
    if (static_cast<int64_t>(raw_values.size()) != params.n) {
        throw std::invalid_argument(
            "build_database_polynomial_eval_form_from_raw_values: raw_values.size() (" +
            std::to_string(raw_values.size()) + ") must equal params.n (" + std::to_string(params.n) + ")");
    }

    FHEDeck::Polynomial poly = build_polynomial_from_raw_values(params, raw_values, modulus);

    std::shared_ptr<FHEDeck::PolynomialEvalForm> poly_eval =
        ctx.rlwe_param->mul_engine()->init_polynomial_eval_form();
    poly.to_eval(*poly_eval, ctx.rlwe_param->mul_engine());

    return DatabasePolynomialEvalForm{std::move(poly_eval), raw_values};
}

DatabasePolynomialEvalForm build_random_database_polynomial_eval_form(const CryptoContext& ctx,
                                                                       const Params& params,
                                                                       std::mt19937_64& rng) {
    std::vector<int64_t> raw_values(static_cast<size_t>(params.n));
    for (int64_t i = 0; i < params.n; ++i) {
        raw_values[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
    }
    return build_database_polynomial_eval_form_from_raw_values(ctx, params, raw_values, params.plaintext_modulus);
}

std::vector<DatabasePolynomialEvalForm> crt_split_database_polynomial_eval_form(const CryptoContext& ctx,
                                                                                  const Params& params,
                                                                                  const std::vector<int64_t>& raw_values) {
    int64_t r = params.num_component_rings;
    std::vector<DatabasePolynomialEvalForm> result;
    result.reserve(static_cast<size_t>(r));
    for (int64_t ring = 0; ring < r; ++ring) {
        result.push_back(build_database_polynomial_eval_form_from_raw_values(
            ctx, params, raw_values, component_ring_modulus(params, ring)));
    }
    return result;
}

FHEDeck::Vector build_vector_from_raw_values(const Params& params, const std::vector<int64_t>& raw_values,
                                              int64_t modulus) {
    if (static_cast<int64_t>(raw_values.size()) != params.n) {
        throw std::invalid_argument("build_vector_from_raw_values: raw_values.size() (" +
                                     std::to_string(raw_values.size()) + ") must equal params.n (" +
                                     std::to_string(params.n) + ")");
    }
    std::vector<int64_t> reduced(static_cast<size_t>(params.n));
    for (int64_t i = 0; i < params.n; ++i) {
        reduced[static_cast<size_t>(i)] = reduce_mod(raw_values[static_cast<size_t>(i)], modulus);
    }
    return FHEDeck::Vector(reduced, params.n, modulus);
}

std::vector<FHEDeck::Vector> crt_split_vector_from_raw_values(const Params& params,
                                                                const std::vector<int64_t>& raw_values) {
    int64_t r = params.num_component_rings;
    std::vector<FHEDeck::Vector> result;
    result.reserve(static_cast<size_t>(r));
    for (int64_t ring = 0; ring < r; ++ring) {
        result.push_back(build_vector_from_raw_values(params, raw_values, component_ring_modulus(params, ring)));
    }
    return result;
}

} // namespace psearch