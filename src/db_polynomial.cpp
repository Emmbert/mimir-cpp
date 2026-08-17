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
    return (max_abs * max_abs) >= ((params.plaintext_modulus + 1) / 2);
}

bool dot_product_can_overflow(const Params& params) {
    __int128 max_abs = max_abs_embedding_value(params);
    // Worst case: every one of embedding_length terms at max magnitude with
    // matching signs, e.g. l * (-8)*(-8).
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

DatabasePolynomialEvalForm build_database_polynomial_eval_form_from_raw_values(const CryptoContext& ctx,
                                                                                const Params& params,
                                                                                const std::vector<int64_t>& raw_values) {
    if (static_cast<int64_t>(raw_values.size()) != params.n) {
        throw std::invalid_argument(
            "build_database_polynomial_eval_form_from_raw_values: raw_values.size() (" +
            std::to_string(raw_values.size()) + ") must equal params.n (" + std::to_string(params.n) + ")");
    }

    FHEDeck::Polynomial poly(params.n, params.q);
    for (int64_t i = 0; i < params.n; ++i) {
        poly[i] = reduce_mod(raw_values[static_cast<size_t>(i)], params.plaintext_modulus);
    }

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
    return build_database_polynomial_eval_form_from_raw_values(ctx, params, raw_values);
}

} // namespace psearch