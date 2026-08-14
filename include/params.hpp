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

    /// Derives cluster_size / splits_per_cluster / clusters_per_server from the
    /// primary parameters (database_size, num_clusters, n, num_servers).
    /// Call this after setting the primary fields and before using the struct anywhere.
    void derive_dependent_parameters();

    /// Convenience factory: small parameters for the correctness tests.
    static Params make_test_params();

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

    /// Builds rlwe_param / both gadgets / encoding from a Params instance.
    static CryptoContext from_params(const Params& params);
};

} // namespace psearch