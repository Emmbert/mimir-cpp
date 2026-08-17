// test_client_server_roundtrip.cpp
//
// The capstone test: a REAL server (background thread, real listening
// socket, real accept loop) and a REAL client (client_core.hpp's functions
// + a real Socket connection -- the exact same code path client.cpp uses)
// talk to each other over 127.0.0.1. This is different from, and stronger
// than, every earlier scoring test: those confirmed "the protocol's result
// matches the plaintext computation AT THE POSITION IT RETURNED." This test
// confirms the position it returned is genuinely the TRUE maximum across
// the entire desired cluster -- computed independently, by scanning every
// split and every position's true dot product directly from the database.
// A bug that made the client return a non-maximal position which still
// happened to be internally consistent with the plaintext at THAT position
// would pass every earlier test but fail this one.
//
// Requires cpp_database_files/test_db_MSMarco_5100_l192_rho4_c2.mdb (or
// whichever file matches Params::make_test_database_params()) -- skips if
// not present.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_client_server_roundtrip

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <future>
#include <random>
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

// Fixed port for this test. A small, accepted risk of collision in CI --
// Socket doesn't currently expose "what port did the OS actually assign me"
// for a bind-to-0 setup, so this is the pragmatic choice for now.
constexpr uint16_t kTestPort = 18734;

/// Runs a minimal server: binds, signals readiness via `ready`, handles
/// exactly two requests (one Registration, one Query) via handle_message,
/// then returns. Meant to run on its own thread for the duration of one
/// test.
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

TEST(ClientServerRoundtrip, ReturnedPositionIsTheTrueMaximumInTheDatabase) {
    if (!std::filesystem::exists(Params::kTestDatabaseFilePath)) {
        GTEST_SKIP() << "Expected test database file at " << Params::kTestDatabaseFilePath
                     << " -- see convert_clusters_to_database.py to generate it.";
    }

    Params params = Params::make_test_database_params();
    ASSERT_FALSE(products_can_overflow(params));
    ASSERT_FALSE(dot_product_can_overflow(params));

    CryptoContext ctx = CryptoContext::from_params(params);
    ServerDatabase db = ServerDatabase::load_from_file(Params::kTestDatabaseFilePath);

    ASSERT_EQ(db.embedding_length(), params.embedding_length)
        << "Database file doesn't match make_test_database_params().";
    ASSERT_EQ(db.num_clusters(), params.num_clusters);
    ASSERT_NO_THROW(validate_value_range(db, params));
    ASSERT_NO_THROW(validate_uniform_cluster_sizes(db, params));

    // --- Start the real server on a background thread. -----------------------
    std::promise<void> ready_promise;
    std::future<void> ready_future = ready_promise.get_future();
    std::thread server_thread(run_test_server, std::cref(ctx), std::cref(params), std::cref(db),
                               std::move(ready_promise));

    ASSERT_EQ(ready_future.wait_for(std::chrono::seconds(5)), std::future_status::ready)
        << "Server thread did not start listening within 5 seconds.";

    // --- Build a real query -- fixed seed, so this test is reproducible. -----
    std::mt19937_64 rng(12345);
    std::vector<SignedValue> query_values;
    std::vector<int64_t> embedding_values;
    query_values.reserve(static_cast<size_t>(params.embedding_length));
    embedding_values.reserve(static_cast<size_t>(params.embedding_length));
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        SignedValue v = sample_signed_value(params, rng);
        query_values.push_back(v);
        embedding_values.push_back(v.reduced);
    }

    // --- Real client, real sockets -- exactly client.cpp's code path. --------
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
        std::vector<uint8_t> query_bytes =
            build_query_message(ctx, registration.session, params, embedding_values, params.desired_cluster_index);
        Socket sock = connect_to_server("127.0.0.1", kTestPort);
        sock.send_message(Message{MessageType::Query, query_bytes});
        Message response = sock.recv_message();
        ASSERT_EQ(response.type, MessageType::QueryResponse)
            << "Query failed: " << std::string(response.payload.begin(), response.payload.end());
        response_bytes = std::move(response.payload);
    }

    ClientQueryResult result = decrypt_and_find_best(ctx, registration.session, params, response_bytes);

    server_thread.join();

    // --- Independently verify: is (split_index, position_in_split)
    // genuinely the TRUE maximum across the whole desired cluster? Scans
    // every split, every position, computes the real dot product directly
    // from the database -- completely independent of anything the
    // client/server pipeline computed. ------------------------------------------
    int64_t true_best_score = std::numeric_limits<int64_t>::min();
    int64_t true_best_split = -1;
    int64_t true_best_position = -1;

    for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
        std::vector<DatabasePolynomialEvalForm> split_data =
            db.build_split(ctx, params, params.desired_cluster_index, s);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t score = 0;
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                score += query_values[static_cast<size_t>(j)].raw *
                         split_data[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            }
            if (score > true_best_score) {
                true_best_score = score;
                true_best_split = s;
                true_best_position = i;
            }
        }
    }

    EXPECT_EQ(result.split_index, true_best_split)
        << "Client returned split " << result.split_index << ", but the true best split (scanned independently) is "
        << true_best_split;
    EXPECT_EQ(result.position_in_split, true_best_position)
        << "Client returned position " << result.position_in_split
        << ", but the true best position (scanned independently) is " << true_best_position;
    EXPECT_EQ(result.score, true_best_score)
        << "Client reported score " << result.score << ", but the true best score (scanned independently) is "
        << true_best_score;
}
