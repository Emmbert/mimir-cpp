#pragma once
#include <ostream>
#include <string>

#include "params.hpp"

namespace psearch {

/// Loads a Params instance from one of the external parameter-search tool's
/// JSON files, mapping its field names onto Params:
///
///   n_LATTICE_DIMENSION  -> n
///   q_CIPHERTEXT_MODULUS -> q
///   sigma_STANDARD_DEV   -> sigma
///   PLAINTEXT_MODULUS    -> plaintext_modulus
///   N_NUM_DOCS           -> database_size
///   l_EMBEDDING_DIM      -> embedding_length
///   EMBEDDING_PRECISION  -> embedding_precision
///   C_NUM_CLUSTERS       -> num_clusters
///
/// num_servers and desired_cluster_index are NOT in these files -- they're
/// deployment/query choices the external tool doesn't optimize over -- so
/// the caller supplies them explicitly (e.g. from a command-line argument).
///
/// IMPORTANT: B_BASE_DIGIT_KS and B_BASE_DIGIT_PRIME in these files are the
/// LITERAL gadget decomposition bases (verified against the files' own
/// d_DECOMP_VEC_LEN_* digit-count fields: digits = ceil(log_B(q))), and the
/// external tool optimizes over arbitrary integer bases -- it does NOT
/// restrict them to powers of two. fhe-deck-core's gadget decomposition only
/// works correctly for powers of two (see Params::derive_dependent_parameters,
/// and the earlier finding that Utils::integer_decomp silently produces
/// wrong results otherwise). So: if a file's base isn't already a power of
/// two, this loader rounds it UP to the nearest power of two and prints a
/// warning showing the substitution. This keeps the file runnable, but means
/// the loaded parameters no longer exactly match that file's own DFR /
/// communication-cost predictions, which assumed the original base. If you'd
/// rather this fail loudly on a non-power-of-two base instead of
/// approximating, see the TODO in params_io.cpp -- it's a one-line change.
///
/// r_NUM_COMP_RINGS is validated to be 1: this codebase doesn't implement
/// CRT decomposition (multiple computation rings) yet, so a file requiring
/// r_NUM_COMP_RINGS > 1 can't be faithfully represented and causes this
/// function to throw.
///
/// Every other field (DFR, LAMBDA, offline/online cost estimates, upload/
/// download sizes, ...) is the external tool's OWN analytical prediction,
/// not something this codebase computes. They're read but not stored in
/// Params -- DFR is printed for reference, so you can compare it against
/// your own measured decryption failure rate.
///
/// Throws std::runtime_error on a missing file, malformed JSON, a missing
/// required field, or r_NUM_COMP_RINGS != 1.
Params load_params_from_json(const std::string& path, int64_t num_servers, int64_t desired_cluster_index = 0);

/// Convenience for benchmark main()s: the ONE shared place every benchmark
/// (benchmark_latency, benchmark_latency_parallel, benchmark_latency_distributed,
/// and any future one) gets its Params from, so they all support the same
/// command-line convention automatically rather than each re-implementing
/// argv parsing:
///
///   ./benchmark_xyz                                  -> Params::make_benchmark_params()
///   ./benchmark_xyz params.json 8                     -> load_params_from_json(path, num_servers=8)
///   ./benchmark_xyz params.json 8 3                   -> also desired_cluster_index=3
///
/// Deliberately NOT used by Params::make_test_params() or any test -- tests
/// need small, fast, hand-verified parameters, and silently loading a
/// realistic (large) parameter file into every correctness test would turn
/// a several-second `ctest` run into minutes/hours. This function exists
/// specifically for benchmark entry points, not general Params construction.
Params load_benchmark_params_from_args(int argc, char** argv);

/// Prints every field of `params` to `os`, plus `source_description` (a
/// file path if one was loaded, or a note that built-in defaults were
/// used). Called by every benchmark, to both stdout and the results file,
/// so a results file is self-describing months later without needing to
/// separately archive the exact command line that produced it.
void print_params(std::ostream& os, const Params& params, const std::string& source_description);

} // namespace psearch