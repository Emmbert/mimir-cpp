// test_client_server_ignores_other_clusters.cpp
//
// Directly tests the property test_client_server_roundtrip.cpp's real-data
// version couldn't actually prove: if the largest matching value in the
// WHOLE database sits in a cluster the client did NOT select, the protocol
// must not return it. That test only checked "the answer matches the true
// max WITHIN the desired cluster" -- which is consistent both with correct
// RGSW masking AND with a world where the real database's global max just
// happens to land inside the desired cluster by luck of the data. This test
// removes that luck: it builds a tiny SYNTHETIC database where cluster 1
// has an unambiguously larger score than anything in cluster 0, queries
// for cluster 0, and confirms the returned result comes from cluster 0's
// (much smaller) data -- not cluster 1's larger one.
//
// Uses the same real-socket client-server round trip as
// test_client_server_roundtrip.cpp, on a DIFFERENT fixed port to avoid
// colliding with it if both run in parallel.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_client_server_ignores_other_clusters

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <thread>
#include <vector>

#include "client_core.hpp"
#include "db_polynomial.hpp"
#include "network.hpp"
#include "params.hpp"
#include "server_db.hpp"
#include "server_dispatch.hpp"
#include "server_session.hpp"

using namespace psearch;

namespace {

constexpr uint16_t kTestPort = 18735; // different from test_client_server_roundtrip.cpp's port

std::filesystem::path write_synthetic_database_file(const std::vector<std::vector<int8_t>>& clusters,
                                                      uint32_t embedding_length) {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / "test_client_server_ignores_other_clusters_tmp.mdb";

    std::ofstream out(path, std::ios::binary);
    uint32_t num_clusters = static_cast<uint32_t>(clusters.size());

    out.write(reinterpret_cast<const char*>(&kServerDatabaseFileMagic), sizeof(kServerDatabaseFileMagic));
    out.write(reinterpret_cast<const char*>(&num_clusters), sizeof(num_clusters));
    out.write(reinterpret_cast<const char*>(&embedding_length), sizeof(embedding_length));

    for (const auto& cluster : clusters) {
        uint32_t num_embeddings = static_cast<uint32_t>(cluster.size() / embedding_length);
        out.write(reinterpret_cast<const char*>(&num_embeddings), sizeof(num_embeddings));
    }
    for (const auto& cluster : clusters) {
        out.write(reinterpret_cast<const char*>(cluster.data()), static_cast<std::streamsize>(cluster.size()));
    }
    return path;
}

void run_test_server(const CryptoContext& ctx, const Params& params, const ServerDatabase& db,
                      std::promise<void> ready) {
    SessionStore sessions;
    Socket listener = create_listening_socket(kTestPort);
    ready.set_value();

    for (int i = 0; i < 2; ++i) {
        Socket client = accept_connection(listener);
        Message request = client.recv_message();
        Message response = handle_message(request, sessions, ctx, params, db);
        client.send_message(response);
    }
}

} // namespace

TEST(ClientServerIgnoresOtherClusters, LargerValueInNonSelectedClusterIsNotReturned) {
    Params params;
    params.n = 2048;
    params.q = 281474976694273; // ~2^48
    params.sigma = 3.2;
    params.plaintext_modulus = 300; // see products_can_overflow/dot_product_can_overflow below:
    // worst case here is 2 * 8*8 = 128, well under 300/2 = 150
    params.decomposition_base_ksk = 16;
    params.decomposition_base_prime = 16;
    params.embedding_length = 2;
    params.embedding_precision = 4; // signed range [-8, 7]
    params.database_size = 2;       // 1 embedding per cluster, 2 clusters
    params.num_clusters = 2;
    params.num_servers = 1;
    params.desired_cluster_index = 0; // querying for cluster 0 specifically
    params.derive_dependent_parameters();


    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);

    // Query = [7, 7] (max magnitude in this precision).
    // Cluster 0 (the one we'll select): embedding = [0, 0] -> true score 0.
    // Cluster 1 (NOT selected): embedding = [7, 7] -> true score 98 --
    // unambiguously the larger value in the WHOLE database.
    std::vector<int8_t> cluster0 = {0, 0};
    std::vector<int8_t> cluster1 = {7, 7};
    auto db_path = write_synthetic_database_file({cluster0, cluster1}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(db_path.string());
    ASSERT_NO_THROW(validate_value_range(db, params));
    ASSERT_NO_THROW(validate_uniform_cluster_sizes(db, params));

    // --- Start the real server. -----------------------------------------------
    std::promise<void> ready_promise;
    std::future<void> ready_future = ready_promise.get_future();
    std::thread server_thread(run_test_server, std::cref(ctx), std::cref(params), std::cref(db),
                               std::move(ready_promise));
    ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "Server thread did not start listening within 5 seconds.";

    std::vector<int64_t> embedding_values = {7, 7};
    for (auto& v : embedding_values) {
        v = reduce_mod(v, params.plaintext_modulus);
    }

    // --- Real client, real sockets, targeting cluster 0. -----------------------
    RegistrationBundle registration = build_registration(ctx, params);
    {
        Socket sock = connect_to_server("127.0.0.1", kTestPort);
        sock.send_message(Message{MessageType::Registration, registration.message_bytes});
        Message response = sock.recv_message();
        ASSERT_EQ(response.type, MessageType::RegistrationAck)
            << "Registration failed: " << std::string(response.payload.begin(), response.payload.end());
    }

    std::vector<uint8_t> response_bytes;
    {
        std::vector<uint8_t> query_bytes = build_query_message(ctx, registration.session, params, embedding_values,
                                                                 /*desired_cluster_index=*/0);
        Socket sock = connect_to_server("127.0.0.1", kTestPort);
        sock.send_message(Message{MessageType::Query, query_bytes});
        Message response = sock.recv_message();
        ASSERT_EQ(response.type, MessageType::QueryResponse)
            << "Query failed: " << std::string(response.payload.begin(), response.payload.end());
        response_bytes = std::move(response.payload);
    }

    ClientQueryResult result = decrypt_and_find_best(ctx, registration.session, params, response_bytes);

    server_thread.join();
    std::filesystem::remove(db_path);

    // --- The actual check: the much larger value (98) sitting in cluster 1
    // must NOT show up, even though it's the true global maximum across
    // the whole database. Cluster 0's true (and only) value here is 0. -----------
    EXPECT_EQ(result.score, 0) << "Expected cluster 0's own value (0), but got " << result.score
                                << " -- if this is 98, cluster 1's larger value leaked through despite not "
                                << "being the selected cluster.";
}
