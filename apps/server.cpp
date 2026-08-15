// server.cpp
//
// Usage:
//   ./server <params.json> <database.mdb> <port>
//
// Setup: loads Params from a JSON parameter file, loads the real database,
// validates it (both value range AND uniform cluster sizes -- see
// server_db.hpp), then listens for connections.
//
// Per connection: reads exactly ONE message, dispatches, responds, closes.
// Registration and Query are NOT required to happen on the same connection
// -- a client registers once, then can reconnect later (up to the session
// TTL) for each subsequent query, looked up by the session ID it kept.
//
//   Registration -> unpack eval keys (reconstruct_public_material, no
//                   secret key involved), store in the session store,
//                   reply RegistrationAck.
//   Query        -> look up the session; if missing/expired, reply Error
//                   asking the client to register again. Otherwise run
//                   process_query (the full multithreaded protocol against
//                   the real database) and reply QueryResponse.
//
// Single-threaded accept loop processing one client at a time: each query
// already uses every core via OpenMP internally, so handling multiple
// clients' queries concurrently would oversubscribe the CPU rather than
// add real throughput. The session store is still built to be thread-safe
// (std::mutex) in case a future version wants concurrent client handling.

#include <iostream>
#include <string>

#include "database_metadata.hpp"
#include "network.hpp"
#include "params.hpp"
#include "params_io.hpp"
#include "server_db.hpp"
#include "server_query_processing.hpp"
#include "server_session.hpp"
#include "wire_protocol.hpp"

using namespace psearch;

namespace {

Message make_error(const std::string& text) {
    Message msg;
    msg.type = MessageType::Error;
    msg.payload.assign(text.begin(), text.end());
    return msg;
}

void handle_registration(const std::vector<uint8_t>& payload, SessionStore& sessions, const CryptoContext& ctx,
                          const Params& params, const Socket& client) {
    RegistrationMessage req = deserialize_registration_message(payload);
    ClientPublicMaterial pub = reconstruct_public_material(ctx, params, req.material);
    sessions.register_session(req.session_id, std::move(pub));

    Message ack;
    ack.type = MessageType::RegistrationAck;
    client.send_message(ack);
}

void handle_query(const std::vector<uint8_t>& payload, SessionStore& sessions, const CryptoContext& ctx,
                   const Params& params, const ServerDatabase& db, const Socket& client) {
    QueryMessage req = deserialize_query_message(payload);

    std::optional<ClientPublicMaterial> pub = sessions.get_and_touch(req.session_id);
    if (!pub.has_value()) {
        client.send_message(make_error("unknown or expired session -- please register again"));
        return;
    }

    QueryResponse response = process_query(ctx, params, db, *pub, req.query);

    Message reply;
    reply.type = MessageType::QueryResponse;
    reply.payload = serialize_query_response(ctx, response);
    client.send_message(reply);
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <params.json> <database.mdb> <port>\n";
        return 1;
    }
    std::string params_path = argv[1];
    std::string database_path = argv[2];
    uint16_t port = static_cast<uint16_t>(std::stoi(argv[3]));

    Params params = load_params_from_json(params_path, /*num_servers=*/1, /*desired_cluster_index=*/0);
    CryptoContext ctx = CryptoContext::from_params(params);

    std::cout << "Loading database from " << database_path << "...\n";
    ServerDatabase db = ServerDatabase::load_from_file(database_path);
    std::cout << "Loaded " << db.num_clusters() << " clusters, embedding_length=" << db.embedding_length() << "\n";

    if (db.embedding_length() != params.embedding_length) {
        std::cerr << "ERROR: database embedding_length (" << db.embedding_length()
                  << ") does not match params.embedding_length (" << params.embedding_length << ")\n";
        return 1;
    }

    try {
        validate_value_range(db, params);
        validate_uniform_cluster_sizes(db, params);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    SessionStore sessions;

    Socket listener = create_listening_socket(port);
    std::cout << "Listening on port " << port << "...\n";

    while (true) {
        Socket client;
        try {
            client = accept_connection(listener);
        } catch (const std::exception& e) {
            std::cerr << "accept failed: " << e.what() << "\n";
            continue;
        }

        try {
            Message request = client.recv_message();

            switch (request.type) {
                case MessageType::Registration:
                    handle_registration(request.payload, sessions, ctx, params, client);
                    break;
                case MessageType::Query:
                    handle_query(request.payload, sessions, ctx, params, db, client);
                    break;
                default:
                    client.send_message(make_error("unexpected message type"));
                    break;
            }
        } catch (const std::exception& e) {
            std::cerr << "error handling client: " << e.what() << "\n";
            try {
                client.send_message(make_error(std::string("server error: ") + e.what()));
            } catch (...) {
                // best-effort -- connection may already be broken
            }
        }
        // Socket destructor closes the connection here.
    }
}
