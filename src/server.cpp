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
#include "server_dispatch.hpp"
#include "server_session.hpp"

using namespace psearch;

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
            Message response = handle_message(request, sessions, ctx, params, db);
            client.send_message(response);
        } catch (const std::exception& e) {
            std::cerr << "error handling client: " << e.what() << "\n";
            try {
                Message err;
                err.type = MessageType::Error;
                std::string text = std::string("server error: ") + e.what();
                err.payload.assign(text.begin(), text.end());
                client.send_message(err);
            } catch (...) {
                // best-effort -- connection may already be broken
            }
        }
        // Socket destructor closes the connection here.
    }
}