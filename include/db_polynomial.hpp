#pragma once
#include <random>
#include <vector>

#include "fhe_deck.h"
#include "params.hpp"

namespace psearch {

/// Lower bound of the signed embedding/database value range:
/// -2^(embedding_precision - 1).
int64_t min_embedding_value(const Params& params);

/// Upper bound of the signed embedding/database value range:
/// 2^(embedding_precision - 1) - 1.
int64_t max_embedding_value(const Params& params);

/// Largest possible absolute value in the signed range, i.e.
/// -min_embedding_value(params) == 2^(embedding_precision - 1). (The lower
/// bound is one larger in magnitude than the upper bound, as usual for a
/// signed range, so this is always the binding one.)
int64_t max_abs_embedding_value(const Params& params);

/// True if two values from this range could multiply to something whose
/// magnitude is >= plaintext_modulus/2 — i.e. something that, once reduced
/// mod plaintext_modulus, can no longer be decoded back to the true signed
/// product via the usual "centered" convention (value if value <=
/// plaintext_modulus/2, else value - plaintext_modulus). This is a
/// worst-case check based on max_abs_embedding_value alone, independent of
/// any particular sampled values.
bool products_can_overflow(const Params& params);

/// Same idea as products_can_overflow, but for a dot product of
/// params.embedding_length terms rather than a single product: checks
/// whether embedding_length * max_abs_embedding_value(params)^2 (the
/// worst case, if every term had the same sign and maximum magnitude) can
/// reach plaintext_modulus/2. Uses 128-bit arithmetic internally so the
/// check itself can't silently overflow before parameters are tuned safely.
bool dot_product_can_overflow(const Params& params);

/// Reduces a signed integer to its canonical non-negative residue mod
/// `modulus`. Plain C++ '%' can return a negative result for a negative
/// dividend, so this corrects for that explicitly.
int64_t reduce_mod(int64_t value, int64_t modulus);

/// Interprets a non-negative residue in [0, modulus) as a signed value via
/// the standard centered convention: unchanged if <= modulus/2, else
/// value - modulus. This is the inverse of reduce_mod ONLY if the original
/// signed value had magnitude < modulus/2 to begin with — if it didn't
/// (overflow), this will return something other than the true value, which
/// is exactly the condition the tests need to detect.
int64_t centered_residue(int64_t value, int64_t modulus);

/// Convenience overload: same as centered_residue(value, params.plaintext_modulus).
/// Named to read as "the inverse of the mod-plaintext_modulus reduction that
/// happened at sampling time" wherever a Params is already in scope.
int64_t decode_to_signed(int64_t value, const Params& params);

/// Vector version of centered_residue: applies it element-wise to every
/// entry of `values` (mod `modulus`), returning a plain std::vector<int64_t>
/// of signed values. Useful for decoding a whole decrypted polynomial's
/// coefficients at once, instead of looping and calling centered_residue
/// per-coefficient inline in test code.
std::vector<int64_t> centered_residues(const FHEDeck::Vector& values, int64_t modulus);

/// Convenience overload: same as centered_residues(values, params.plaintext_modulus).
std::vector<int64_t> decode_to_signed(const FHEDeck::Vector& values, const Params& params);

/// One sampled value, kept in both forms:
///   - raw: the true signed integer, e.g. in [-8, 7] for embedding_precision 4.
///     Never touches plaintext_modulus. Use this for computing the true,
///     un-reduced mathematical result to check against for overflow.
///   - reduced: raw's canonical non-negative residue mod plaintext_modulus.
///     This is what actually gets encrypted / used as a polynomial
///     coefficient, since FHE-Deck's encode_message/decode_message operate
///     on non-negative residues, not signed integers.
struct SignedValue {
    int64_t raw;
    int64_t reduced;
};

/// Draws one signed value uniformly from [min_embedding_value, max_embedding_value].
SignedValue sample_signed_value(const Params& params, std::mt19937_64& rng);

/// Convenience wrapper around sample_signed_value for callers that only need
/// the reduced (encryptable) form, e.g. the LWE->RLWE round-trip test.
int64_t sample_signed_mod_value(const Params& params, std::mt19937_64& rng);

/// One randomly-built database polynomial, kept in both forms for the same
/// reason as SignedValue:
///   - poly: coefficient-form Polynomial with reduced (mod plaintext_modulus)
///     coefficients — what actually gets multiplied against an RLWE
///     ciphertext.
///   - raw_values: the true signed coefficients, same length as poly, for
///     computing the true un-reduced product when checking for overflow.
struct DatabasePolynomial {
    FHEDeck::Polynomial poly;
    std::vector<int64_t> raw_values;
};

/// Builds one DatabasePolynomial of degree params.n, with every coefficient
/// drawn independently via sample_signed_value(...).
///
/// This is deliberately factored out here rather than inlined into a test:
/// it's the exact per-polynomial construction ServerDatabase::build (Step 1
/// of the protocol, still a TODO in server_db.cpp) will use for every
/// polynomial in every split of every cluster.
DatabasePolynomial build_random_database_polynomial(const Params& params, std::mt19937_64& rng);

/// Builds a coefficient-form Polynomial (mod q, matching the RLWE ring)
/// from `raw_values`, each entry reduced mod `modulus`. Generalizes
/// build_random_database_polynomial's `poly` field -- which always reduced
/// against params.plaintext_modulus specifically -- to an arbitrary
/// modulus, so the SAME function covers both the non-CRT case
/// (modulus = params.plaintext_modulus, reproducing the original behavior
/// exactly) and each CRT component ring (modulus = p1 or p2). Throws if
/// raw_values.size() != params.n.
FHEDeck::Polynomial build_polynomial_from_raw_values(const Params& params, const std::vector<int64_t>& raw_values,
                                                      int64_t modulus);

/// Which modulus CRT component `ring` uses (0 or 1). For
/// num_component_rings == 1, ring must be 0 and this returns
/// plaintext_modulus. For num_component_rings == 2, ring 0 returns
/// comp_ring_modulus (p1), ring 1 returns comp_ring_modulus - 1 (p2).
/// Throws on an out-of-range ring.
int64_t component_ring_modulus(const Params& params, int64_t ring);

/// Splits a database polynomial's raw_values into its CRT component-ring
/// coefficient-form polynomials -- one entry per params.num_component_rings,
/// each built via build_polynomial_from_raw_values with that ring's own
/// modulus (see component_ring_modulus). For num_component_rings == 1,
/// returns a single-entry vector, identical to what
/// build_random_database_polynomial's own `poly` field already contains --
/// no separate CRT/non-CRT code path needed by callers.
std::vector<FHEDeck::Polynomial> crt_split_database_polynomial(const Params& params,
                                                                 const std::vector<int64_t>& raw_values);

/// Same as DatabasePolynomial, but the plaintext polynomial is stored
/// directly in NTT/eval form rather than coefficient form — the coefficient
/// form is only ever a transient intermediate used to sample values, never
/// exposed to the caller. This is what the real database will eventually
/// store (Step 1 of the protocol: "All polynomials are transferred into NTT
/// representation") — once ServerDatabase::build exists, it calls this
/// function (or one shaped just like it) for every polynomial in every split
/// of every cluster, so the database is built in eval form from the start
/// rather than converted at query time.
struct DatabasePolynomialEvalForm {
    std::shared_ptr<FHEDeck::PolynomialEvalForm> poly_eval;
    std::vector<int64_t> raw_values; // true signed samples, for overflow-checking in tests
};

/// Builds one DatabasePolynomialEvalForm from EXPLICIT raw signed
/// coefficients (length params.n): reduces each mod `modulus` and converts
/// to eval form. `modulus` generalizes what used to be hardcoded to
/// plaintext_modulus -- pass params.plaintext_modulus to reproduce the
/// original non-CRT behavior exactly, or a CRT component ring's own p1/p2
/// (see component_ring_modulus) to build that ring's polynomial.
///
/// This is the actual construction step both build_random_database_
/// polynomial_eval_form (random sampling) and ServerDatabase::build_split
/// (real loaded data, in the currently-paused server-client layer -- its
/// call site still needs updating to pass params.plaintext_modulus
/// explicitly once that layer's own CRT work resumes) go through --
/// factored out here so every path shares one implementation and can't
/// silently drift apart from each other.
/// @throws std::invalid_argument if raw_values.size() != params.n.
DatabasePolynomialEvalForm build_database_polynomial_eval_form_from_raw_values(const CryptoContext& ctx,
                                                                                const Params& params,
                                                                                const std::vector<int64_t>& raw_values,
                                                                                int64_t modulus);

/// Builds one DatabasePolynomialEvalForm of degree params.n, reduced mod
/// params.plaintext_modulus (the non-CRT case -- see
/// crt_split_database_polynomial_eval_form for the CRT-aware version).
/// Requires a CryptoContext (specifically its rlwe_param->mul_engine()) to
/// perform the coefficient-form -> eval-form conversion internally.
DatabasePolynomialEvalForm build_random_database_polynomial_eval_form(const CryptoContext& ctx,
                                                                       const Params& params,
                                                                       std::mt19937_64& rng);

/// Eval-form counterpart of crt_split_database_polynomial: splits a
/// database polynomial's raw_values into its CRT component-ring EVAL-FORM
/// polynomials -- one entry per params.num_component_rings, each built via
/// build_database_polynomial_eval_form_from_raw_values with that ring's own
/// modulus. For num_component_rings == 1, returns a single-entry vector,
/// identical to build_random_database_polynomial_eval_form's own output --
/// no separate CRT/non-CRT code path needed by callers.
std::vector<DatabasePolynomialEvalForm> crt_split_database_polynomial_eval_form(const CryptoContext& ctx,
                                                                                  const Params& params,
                                                                                  const std::vector<int64_t>& raw_values);

/// Vector counterpart of build_polynomial_from_raw_values, for the
/// RLWESK::encode_and_encrypt(Vector, encoding) API used by dense-message
/// (RGSW-multiplied) tests rather than the coefficient-form Polynomial
/// multiplication used elsewhere. Same modulus-generalization reasoning:
/// pass params.plaintext_modulus for the non-CRT case, or a CRT component
/// ring's own p1/p2 for that ring's vector.
FHEDeck::Vector build_vector_from_raw_values(const Params& params, const std::vector<int64_t>& raw_values,
                                              int64_t modulus);

/// Splits raw_values into its CRT component-ring Vectors -- one entry per
/// params.num_component_rings, each built via build_vector_from_raw_values
/// with that ring's own modulus. For num_component_rings == 1, returns a
/// single-entry vector using plaintext_modulus -- no separate CRT/non-CRT
/// code path needed by callers.
std::vector<FHEDeck::Vector> crt_split_vector_from_raw_values(const Params& params,
                                                                const std::vector<int64_t>& raw_values);

} // namespace psearch