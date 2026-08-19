#pragma once
#include <cstdint>
#include <memory>
#include <vector>

// FHE-Deck core types we build our RLWE ring / gadgets from.
#include "fhe_deck.h"

namespace psearch {

/// Step 0: every configurable knob of the protocol lives here.
/// Kept as a plain struct (no logic) so tests and benchmarks can both build one from
/// scratch, and so we never have "hidden" global parameters like the toy_workflow's
/// parameters.py module-level constants.
///
/// Query/database values are always drawn uniformly at random from the signed
/// embedding range [-2^(embedding_precision-1), 2^(embedding_precision-1)-1],
/// mod-reduced to a canonical residue in [0, plaintext_modulus) -- see
/// sample_signed_value / build_random_database_polynomial_eval_form in
/// db_polynomial.hpp. There used to be a FillMode (Zeros/Ones/Random) here to
/// distinguish deterministic test values from random benchmark values, but
/// every test and the benchmark already used the same random sampling
/// regardless of that field -- it was never actually read anywhere. Removed
/// rather than left as dead, misleading state.
struct Params {
    // ---- Crypto parameters ----------------------------------------------
    int64_t n = 0;                 ///< ring/LWE dimension, must be a power of 2
    int64_t q = 0;                 ///< ciphertext modulus
    double sigma = 0.0;            ///< stddev of the discrete Gaussian error
    int64_t plaintext_modulus = 0; ///< plaintext space size (p)
    int64_t decomposition_base_ksk = 0;   ///< gadget decomposition base for the LWE->RLWE
                                           ///< key-switching key (and the RLWE-side gadget
                                           ///< the LWE->RGSW switch also builds on)
    int64_t decomposition_base_prime = 0; ///< gadget decomposition base for the LWE'
                                           ///< ciphertexts (LWEGadgetSK / LWEGadgetCT) used
                                           ///< to encrypt the cluster-selector bits before
                                           ///< the LWE->RGSW switch

    // ---- Database / protocol shape --------------------------------------
    int64_t database_size = 0;     ///< N, total number of DB entries
    int64_t embedding_length = 0;  ///< l, number of polynomials per cluster/document
    int64_t embedding_precision = 0; ///< bit-width of one embedding/db coefficient;
                                      ///< values are drawn from
                                      ///< [-2^(embedding_precision-1), 2^(embedding_precision-1)-1]
    int64_t num_clusters = 0;      ///< c
    int64_t cluster_size = 0;      ///< N / c  (int division)
    int64_t splits_per_cluster = 0;///< s = ceil(cluster_size / n)
    int64_t num_servers = 0;
    int64_t clusters_per_server = 0; ///< c / num_servers (int division)
    int64_t desired_cluster_index = 0; ///< which cluster the client actually wants

    // ---- CRT / multiple component rings ----------------------------------
    int64_t num_component_rings = 1; ///< r, number of CRT component rings.
                                      ///< 1 = no CRT (plaintext_modulus used directly,
                                      ///< as everywhere in this codebase historically).
                                      ///< 2 = CRT decomposition into two coprime rings,
                                      ///< Z_p1 x Z_p2 with p2 = comp_ring_modulus - 1
                                      ///< (consecutive integers, always coprime -- see
                                      ///< crt.hpp). No other value is currently supported.
    int64_t comp_ring_modulus = 0;   ///< p1, only meaningful when num_component_rings == 2.
                                      ///< p2 is always comp_ring_modulus - 1.
    int64_t combined_component_ring_modulus = 0; ///< p1*(p1-1), cached by
                                      ///< derive_dependent_parameters() (same pattern as
                                      ///< cluster_size/splits_per_cluster/clusters_per_server
                                      ///< -- computed once from the primary fields, not
                                      ///< recomputed on every access). Only meaningful when
                                      ///< num_component_rings == 2; 0 otherwise. Must be >=
                                      ///< plaintext_modulus, validated in
                                      ///< derive_dependent_parameters() -- plaintext_modulus's
                                      ///< meaning is UNCHANGED by CRT, it's still the required
                                      ///< lower bound on the message space size; CRT is an
                                      ///< implementation detail of how that space is
                                      ///< represented, not a different requirement.

    /// Derives cluster_size / splits_per_cluster / clusters_per_server from the
    /// primary parameters (database_size, num_clusters, n, num_servers). Also
    /// validates num_component_rings/comp_ring_modulus when CRT is in use.
    /// Call this after setting the primary fields and before using the struct anywhere.
    void derive_dependent_parameters();

    /// Convenience factory: small parameters for the correctness tests.
    static Params make_test_params();

    static constexpr const char* kTestDatabaseFilePath = "../cpp_database_files/test_db_MSMarco_5100_l10_rho2_c2.mdb";

    static Params make_test_database_params();

    static Params make_test_database_params_with_splits();

    /// Convenience factory: small parameters for CRT (two-component-ring)
    /// correctness tests, mirroring make_test_params()'s hand-picked style
    /// and n/q/sigma/decomposition bases/database_size/embedding fields
    /// exactly, so a non-CRT and CRT test can be directly compared on
    /// otherwise-identical parameters. Same MIMIR_TEST_PARAMS_FILE opt-in as
    /// make_test_params() -- if set, loads from that file instead (and may
    /// have num_component_rings == 1, in which case CRT-specific tests
    /// should skip rather than run against non-CRT parameters).
    static Params make_test_params_component_rings();

    /// Convenience factory: realistic-size parameters for the latency benchmark.
    static Params make_benchmark_params();
};

/// Shared FHE-Deck ring/gadget context, built once from a Params instance and handed
/// (as shared_ptr, matching fhe-deck-core's own convention) to client, server and
/// worker code. Keeping this separate from Params means Params stays a plain,
/// copyable value type while this holds the actual library objects.
struct CryptoContext {
    std::shared_ptr<const FHEDeck::RLWEParam> rlwe_param;
    // Two separate gadgets, deliberately NOT one shared object:
    //   gadget_ksk:  used only for LWEToRLWEKeySwitchKey's automorphism
    //                keys. Base = decomposition_base_ksk.
    //   gadget_rgsw: used for the RGSW ciphertext's own internal structure
    //                (message*sk row, and later decomposing incoming
    //                ciphertexts in RLWEGadgetCT::mul). Base =
    //                decomposition_base_prime -- MUST match whatever base
    //                the client's LWEGadgetCT (LWE') was built with.
    // See LWEToRGSWKeySwitchKey's two-argument constructor for why these
    // can't be the same object: reusing one gadget for both jobs silently
    // couples decomposition_base_ksk and decomposition_base_prime together,
    // even though they serve genuinely unrelated purposes.
    std::shared_ptr<FHEDeck::SignedDecompositionGadget> gadget_ksk;
    std::shared_ptr<FHEDeck::SignedDecompositionGadget> gadget_rgsw;
    FHEDeck::PlaintextEncoding encoding{FHEDeck::PlaintextEncodingType::full_domain, 0, 0};

    // One encoding per CRT component ring, built from p1=comp_ring_modulus
    // (and p2=comp_ring_modulus-1 when num_component_rings==2), sharing q.
    // When num_component_rings == 1, this is a single-entry vector holding
    // exactly `encoding` above -- CRT-aware code (build_seeded_query,
    // reconstruct_query, and anything downstream) can always iterate over
    // component_encodings without a separate CRT/non-CRT branch; the
    // num_component_rings==1 case falls out automatically as a
    // length-1 loop. `encoding` itself is kept unchanged and untouched by
    // CRT -- every existing non-CRT code path keeps using it exactly as
    // before.
    std::vector<FHEDeck::PlaintextEncoding> component_encodings;

    /// Builds rlwe_param / both gadgets / encoding / component_encodings
    /// from a Params instance.
    static CryptoContext from_params(const Params& params);
};

} // namespace psearch