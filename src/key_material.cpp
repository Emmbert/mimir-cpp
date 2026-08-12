#include "key_material.hpp"

using namespace FHEDeck;

namespace psearch {

    ClientSecretMaterial generate_client_secret_material(const CryptoContext& ctx, const Params& params) {
        auto rlwe_sk = std::make_shared<RLWESK>(ctx.rlwe_param, KeyDistribution::ternary, params.sigma);
        auto rlwe_gadget_sk = std::make_shared<RLWEGadgetSK>(ctx.gadget, rlwe_sk);
        std::shared_ptr<LWESK> lwe_sk = rlwe_sk->extract_lwe_key();
        auto lwe_gadget_sk = std::make_shared<LWEGadgetSK>(lwe_sk, params.decomposition_base_prime);

        return ClientSecretMaterial{rlwe_sk, rlwe_gadget_sk, lwe_sk, lwe_gadget_sk};
    }

    ClientPublicMaterial generate_client_public_material(const CryptoContext& ctx,
                                                           const ClientSecretMaterial& secret) {
        auto lwe_to_rlwe_ksk =
            std::make_shared<LWEToRLWEKeySwitchKey>(*secret.lwe_sk, *secret.rlwe_gadget_sk);
        auto lwe_to_rgsw_ksk =
            std::make_shared<LWEToRGSWKeySwitchKey>(*secret.lwe_sk, *secret.rlwe_gadget_sk);

        return ClientPublicMaterial{lwe_to_rlwe_ksk, lwe_to_rgsw_ksk};
    }

} // namespace psearch