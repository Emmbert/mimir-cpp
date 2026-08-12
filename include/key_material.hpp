#pragma once
#include <memory>

#include "fhe_deck.h"
#include "params.hpp"

namespace psearch {

// ============================================================================
// Three categories of crypto material, matching who owns/sees each one:
//
//   1. Global   -> psearch::CryptoContext (already in params.hpp): rlwe_param,
//                  gadget, encoding. Same for every client and the server;
//                  not secret, not client-specific, built once from Params.
//   2. Secret   -> ClientSecretMaterial: never leaves the client.
//   3. Public   -> ClientPublicMaterial: exactly what the client sends to the
//                  server during registration (Step 2/3 of the protocol).
//                  Grows over time (currently just the LWE->RLWE key-switch
//                  key; the LWE->RGSW key-switch key for cluster selection
//                  will join it once that part of the protocol is tested).
// ============================================================================

struct ClientSecretMaterial {
    std::shared_ptr<FHEDeck::RLWESK> rlwe_sk;
    std::shared_ptr<FHEDeck::RLWEGadgetSK> rlwe_gadget_sk;
    std::shared_ptr<FHEDeck::LWESK> lwe_sk;
    // Needed to gadget-encrypt LWE messages (LWEGadgetCT) — the input format
    // the LWE->RGSW key switch expects. Used for the cluster-selection unit
    // vector in Step 4 of the protocol.
    std::shared_ptr<FHEDeck::LWEGadgetSK> lwe_gadget_sk;
};

struct ClientPublicMaterial {
    std::shared_ptr<FHEDeck::LWEToRLWEKeySwitchKey> lwe_to_rlwe_ksk;
    // Converts a client's gadget-encrypted LWE ciphertext (LWEGadgetCT) into
    // an RGSW ciphertext (RLWEGadgetCT) — this is what the server uses on
    // the cluster-selection unit vector in Step 5.
    std::shared_ptr<FHEDeck::LWEToRGSWKeySwitchKey> lwe_to_rgsw_ksk;
};

/// Generates fresh secret keys for one client, against the given global
/// CryptoContext. This is Step 2a of the protocol.
ClientSecretMaterial generate_client_secret_material(const CryptoContext& ctx, const Params& params);

/// Builds the public (server-facing) key-switching material from a client's
/// secret material: both the LWE->RLWE key-switching key (Step 2b/3) and the
/// LWE->RGSW key-switching key (used for the cluster-selection unit vector).
/// Takes the CryptoContext too since key switches are built against its
/// rlwe_param/gadget.
ClientPublicMaterial generate_client_public_material(const CryptoContext& ctx,
                                                       const ClientSecretMaterial& secret);

} // namespace psearch