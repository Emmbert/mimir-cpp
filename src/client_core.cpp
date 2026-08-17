#include "client_core.hpp"

#include <limits>

#include "db_polynomial.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"

using namespace FHEDeck;

namespace psearch {

RegistrationBundle build_registration(const CryptoContext& ctx, const Params& params) {
    RegistrationBundle bundle;
    bundle.session.session_id = generate_fresh_seed(); // same 16-byte convention as every other seed
    bundle.session.secret = generate_client_secret_material(ctx, params);

    SeededClientPublicMaterial eval_wire = build_seeded_public_material(ctx, bundle.session.secret);

    RegistrationMessage msg{bundle.session.session_id, eval_wire};
    bundle.message_bytes = serialize_registration_message(msg);
    return bundle;
}

std::vector<uint8_t> build_query_message(const CryptoContext& ctx, ClientSession& session, const Params& params,
                                          const std::vector<int64_t>& embedding_values,
                                          int64_t desired_cluster_index) {
    SeededQuery query_wire = build_seeded_query(ctx, session.secret, embedding_values, params.num_clusters,
                                                 desired_cluster_index);

    QueryMessage msg{session.session_id, query_wire};
    return serialize_query_message(msg);
}

ClientQueryResult decrypt_and_find_best(const CryptoContext& ctx, const ClientSession& session, const Params& params,
                                         const std::vector<uint8_t>& response_bytes) {
    QueryResponse response = deserialize_query_response(ctx, response_bytes);

    int64_t best_score = std::numeric_limits<int64_t>::min();
    int64_t best_split = -1;
    int64_t best_position = -1;

    for (size_t s = 0; s < response.ciphertexts.size(); ++s) {
        Vector decrypted = session.secret.rlwe_sk->decrypt_vector(response.ciphertexts[s], ctx.encoding);
        std::vector<int64_t> decoded_signed = decode_to_signed(decrypted, params);

        for (int64_t i = 0; i < params.n; ++i) {
            if (decoded_signed[static_cast<size_t>(i)] > best_score) {
                best_score = decoded_signed[static_cast<size_t>(i)];
                best_split = static_cast<int64_t>(s);
                best_position = i;
            }
        }
    }

    return ClientQueryResult{best_split, best_position, best_score};
}

} // namespace psearch
