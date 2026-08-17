// client.cpp
//
// Usage:
//   ./client <params.json> <server_host> <server_port> [desired_cluster_index]
//
// Uses client_core.hpp's transport-agnostic functions (build_registration,
// build_query_message, decrypt_and_find_best) together with network.hpp's
// Socket for the actual connection. This split is deliberate: client_core
// itself has zero networking dependency, specifically so a later
// browser/WASM entry point can reuse it unchanged, with JS handling the
// actual transport (fetch/WebSocket) instead of this file's Socket calls.
//
// FOR NOW: generates a RANDOM embedding query (matching
// params.embedding_length/embedding_precision) rather than a real
// HuggingFace-derived one. See the single TODO block below for exactly
// where the real pipeline plugs in later:
//   1. sentence-transformer encode (HuggingFace, Python)
//   2. PCA-reduce using the SAME components/mean the database was reduced with
//   3. quantize to embedding_precision-bit integers using the SAME scale
//      the database was quantized with (see database_metadata.hpp's
//      DatabaseMetadata::scale -- the sidecar written by
//      convert_clusters_to_database.py)
//   4. nearest-centroid lookup (against public centroid data) to pick
//      desired_cluster_index
// None of that is wired up yet, by design -- validating the protocol
// end-to-end with random vectors first, per the earlier discussion.
//
// desired_cluster_index defaults to 0 if not given on the command line.

#include <iostream>
#include <random>
#include <string>

#include "client_core.hpp"
#include "db_polynomial.hpp"
#include "network.hpp"
#include "params.hpp"
#include "params_io.hpp"

using namespace psearch;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <params.json> <server_host> <server_port> [desired_cluster_index]\n";
        return 1;
    }
    std::string params_path = argv[1];
    std::string server_host = argv[2];
    uint16_t server_port = static_cast<uint16_t>(std::stoi(argv[3]));
    int64_t desired_cluster_index = (argc >= 5) ? std::stoll(argv[4]) : 0;

    Params params = load_params_from_json(params_path, /*num_servers=*/1, desired_cluster_index);
    CryptoContext ctx = CryptoContext::from_params(params);

    // --- TODO: replace with a real query embedding -- see file header. --------
    std::mt19937_64 rng(std::random_device{}());
    std::vector<int64_t> embedding_values;
    embedding_values.reserve(static_cast<size_t>(params.embedding_length));
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        embedding_values.push_back(sample_signed_mod_value(params, rng));
    }
    // --- end TODO block. ---------------------------------------------------------

    std::cout << "Registering with " << server_host << ":" << server_port << "...\n";
    RegistrationBundle registration = build_registration(ctx, params);
    {
        Socket sock = connect_to_server(server_host, server_port);
        sock.send_message(Message{MessageType::Registration, registration.message_bytes});

        Message response = sock.recv_message();
        if (response.type == MessageType::Error) {
            std::cerr << "Registration failed: " << std::string(response.payload.begin(), response.payload.end())
                      << "\n";
            return 1;
        }
        if (response.type != MessageType::RegistrationAck) {
            std::cerr << "Registration failed: unexpected response type\n";
            return 1;
        }
    }
    std::cout << "Registered.\n";

    std::cout << "Querying (desired_cluster_index=" << desired_cluster_index << ")...\n";
    std::vector<uint8_t> query_bytes =
        build_query_message(ctx, registration.session, params, embedding_values, desired_cluster_index);

    std::vector<uint8_t> response_bytes;
    {
        Socket sock = connect_to_server(server_host, server_port);
        sock.send_message(Message{MessageType::Query, query_bytes});

        Message response = sock.recv_message();
        if (response.type == MessageType::Error) {
            std::cerr << "Query failed: " << std::string(response.payload.begin(), response.payload.end()) << "\n";
            return 1;
        }
        if (response.type != MessageType::QueryResponse) {
            std::cerr << "Query failed: unexpected response type\n";
            return 1;
        }
        response_bytes = std::move(response.payload);
    }

    ClientQueryResult result = decrypt_and_find_best(ctx, registration.session, params, response_bytes);

    int64_t global_document_index = result.split_index * params.n + result.position_in_split;

    std::cout << "\nBest match:\n";
    std::cout << "  cluster:                " << desired_cluster_index << "\n";
    std::cout << "  split:                  " << result.split_index << "\n";
    std::cout << "  position within split:  " << result.position_in_split << "\n";
    std::cout << "  document index (global, within cluster): " << global_document_index << "\n";
    std::cout << "  score:                  " << result.score << "\n";

    // Future work: retrieve the actual document at (desired_cluster_index,
    // global_document_index) via PIR -- not implemented yet.

    return 0;
}
