#include "key_material.hpp"

using namespace FHEDeck;

namespace psearch {

ClientSecretMaterial generate_client_secret_material(const CryptoContext& ctx, const Params& params) {
    auto rlwe_sk = std::make_shared<RLWESK>(ctx.rlwe_param, KeyDistribution::ternary, params.sigma);

    // Two gadget secret keys, same rlwe_sk, different gadgets -- see
    // CryptoContext and ClientSecretMaterial for why these are kept separate.
    auto rlwe_gadget_sk_ksk = std::make_shared<RLWEGadgetSK>(ctx.gadget_ksk, rlwe_sk);
    auto rlwe_gadget_sk_rgsw = std::make_shared<RLWEGadgetSK>(ctx.gadget_rgsw, rlwe_sk);

    std::shared_ptr<LWESK> lwe_sk = rlwe_sk->extract_lwe_key();

    // LWE' ciphertexts (cluster-selector bits, before the LWE->RGSW switch)
    // use decomposition_base_prime -- MUST match ctx.gadget_rgsw's base,
    // since LWEToRGSWKeySwitchKey treats each row of the resulting
    // LWEGadgetCT as one row of the output RGSW ciphertext's message row,
    // and that row must agree with ct_of_sk_dest's own base (built from
    // rlwe_gadget_sk_rgsw) for RLWEGadgetCT::mul to decompose correctly
    // later.
    auto lwe_gadget_sk = std::make_shared<LWEGadgetSK>(lwe_sk, params.decomposition_base_prime);

    return ClientSecretMaterial{rlwe_sk, rlwe_gadget_sk_ksk, rlwe_gadget_sk_rgsw, lwe_sk, lwe_gadget_sk};
}

ClientPublicMaterial generate_client_public_material(const CryptoContext& ctx,
                                                       const ClientSecretMaterial& secret) {
    auto lwe_to_rlwe_ksk =
        std::make_shared<LWEToRLWEKeySwitchKey>(*secret.lwe_sk, *secret.rlwe_gadget_sk_ksk);

    // Two-gadget constructor: ksk gadget for the plain per-digit LWE->RLWE
    // switch, rgsw gadget for the message*sk row and the output ciphertext's
    // own decomposition base.
    auto lwe_to_rgsw_ksk = std::make_shared<LWEToRGSWKeySwitchKey>(
        *secret.lwe_sk, *secret.rlwe_gadget_sk_ksk, *secret.rlwe_gadget_sk_rgsw);

    return ClientPublicMaterial{lwe_to_rlwe_ksk, lwe_to_rgsw_ksk};
}

} // namespace psearch