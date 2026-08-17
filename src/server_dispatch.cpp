#include "server_dispatch.hpp"

#include "seeded_eval_keys.hpp"
#include "server_query_processing.hpp"

namespace psearch {

namespace {

Message make_error(const std::string& text) {
    Message msg;
    msg.type = MessageType::Error;
    msg.payload.assign(text.begin(), text.end());
    return msg;
}

} // namespace

Message handle_message(const Message& request, SessionStore& sessions, const CryptoContext& ctx,
                        const Params& params, const ServerDatabase& db) {
    try {
        switch (request.type) {
            case MessageType::Registration: {
                RegistrationMessage req = deserialize_registration_message(request.payload);
                ClientPublicMaterial pub = reconstruct_public_material(ctx, params, req.material);
                sessions.register_session(req.session_id, std::move(pub));
                return Message{MessageType::RegistrationAck, {}};
            }
            case MessageType::Query: {
                QueryMessage req = deserialize_query_message(request.payload);
                std::optional<ClientPublicMaterial> pub = sessions.get_and_touch(req.session_id);
                if (!pub.has_value()) {
                    return make_error("unknown or expired session -- please register again");
                }
                QueryResponse response = process_query(ctx, params, db, *pub, req.query);
                return Message{MessageType::QueryResponse, serialize_query_response(ctx, response)};
            }
            default:
                return make_error("unexpected message type");
        }
    } catch (const std::exception& e) {
        return make_error(std::string("server error: ") + e.what());
    }
}

} // namespace psearch
