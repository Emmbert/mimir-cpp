cmake_minimum_required(VERSION 3.16)
project(mimir-cpp CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Set the same options fhe-deck-core's CMakeLists.txt expects
set(USE_EXAMPLES TRUE CACHE BOOL "" FORCE)
set(NTT_ENGINE Hexl CACHE STRING "" FORCE)
set(USE_CEREAL FALSE CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../fhe-deck-core
        ${CMAKE_CURRENT_BINARY_DIR}/fhe-deck-core)

# --- Testing (GoogleTest via FetchContent) ---------------------------------
# No manual install needed: CMake downloads and builds GoogleTest itself.
include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
)
# Keep GoogleTest's runtime settings from clashing with ours on Windows;
# harmless no-op on Linux/macOS.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# --- JSON parsing (nlohmann::json) for loading benchmark parameters -------
FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# --- Core library ------------------------------------------------------------
add_library(mimir_core
        src/params.cpp
        src/key_material.cpp
        src/db_polynomial.cpp
        src/params_io.cpp
        src/seeded_distribution.cpp
        src/seeded_eval_keys.cpp
        src/seeded_query.cpp
        src/server_db.cpp
        src/database_metadata.cpp
        src/server_query_processing.cpp
        src/server_session.cpp
        src/wire_protocol.cpp
        src/network.cpp
        #src/timing.cpp
        #src/channel.cpp
        #src/server_db.cpp
        #src/client.cpp
        #src/server.cpp
)
target_include_directories(mimir_core PUBLIC include)

# Link against whatever target name fhe-deck-core's CMakeLists.txt
# actually defines with add_library(...) — check its CMakeLists.txt,
# it's probably something like "fhe_deck" (check the .a filename too:
# libfhe_deck.a → likely target name "fhe_deck")
target_link_libraries(mimir_core PUBLIC fhe_deck)
target_link_libraries(mimir_core PRIVATE nlohmann_json::nlohmann_json)

# seed expansion via xof from OpenSSL
find_package(OpenSSL REQUIRED)
target_link_libraries(mimir_core PRIVATE OpenSSL::Crypto)

# --- Plain (non-gtest) executables -------------------------------------------
add_executable(kamils_test tests/mimir_performance_tests.cpp)
target_link_libraries(kamils_test PRIVATE mimir_core)

# --- GoogleTest-based unit tests ---------------------------------------------

add_executable(test_lwe_to_rlwe_roundtrip tests/test_lwe_to_rlwe_roundtrip.cpp)
target_link_libraries(test_lwe_to_rlwe_roundtrip PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_lwe_to_rlwe_roundtrip)

add_executable(test_rlwe_plaintext_multiplication tests/test_rlwe_plaintext_multiplication.cpp)
target_link_libraries(test_rlwe_plaintext_multiplication PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_rlwe_plaintext_multiplication
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_rlwe_plaintext_multiplication_ntt tests/test_rlwe_plaintext_multiplication_ntt.cpp)
target_link_libraries(test_rlwe_plaintext_multiplication_ntt PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_rlwe_plaintext_multiplication_ntt
        DISCOVERY_MODE PRE_TEST
)
add_executable(test_rlwe_scoring_dot_product tests/test_rlwe_scoring_dot_product.cpp)
target_link_libraries(test_rlwe_scoring_dot_product PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_rlwe_scoring_dot_product
        DISCOVERY_MODE PRE_TEST
)
add_executable(test_rlwe_scoring_dot_product_coef_sum tests/test_rlwe_scoring_dot_product_coef_sum.cpp)
target_link_libraries(test_rlwe_scoring_dot_product_coef_sum PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_rlwe_scoring_dot_product_coef_sum
        DISCOVERY_MODE PRE_TEST
)
add_executable(test_rgsw_multiplication tests/test_rgsw_multiplication.cpp)
target_link_libraries(test_rgsw_multiplication PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_rgsw_multiplication
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_full_scoring_with_cluster_selection tests/test_full_scoring_with_cluster_selection.cpp)
target_link_libraries(test_full_scoring_with_cluster_selection PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_full_scoring_with_cluster_selection
        DISCOVERY_MODE PRE_TEST
)
add_executable(test_full_scoring_with_splits tests/test_full_scoring_with_splits.cpp)
target_link_libraries(test_full_scoring_with_splits PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_full_scoring_with_splits
        DISCOVERY_MODE PRE_TEST
)
add_executable(test_full_scoring_with_splits_parallel tests/test_full_scoring_with_splits_parallel.cpp)
target_link_libraries(test_full_scoring_with_splits_parallel PRIVATE
        mimir_core
        GTest::gtest_main
        OpenMP::OpenMP_CXX
)
gtest_discover_tests(test_full_scoring_with_splits_parallel
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_seeded_lwe_to_rlwe_roundtrip tests/test_seeded_lwe_to_rlwe_roundtrip.cpp)
target_link_libraries(test_seeded_lwe_to_rlwe_roundtrip PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_seeded_lwe_to_rlwe_roundtrip
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_seeded_key_material_and_query_roundtrip tests/test_seeded_key_material_and_query_roundtrip.cpp)
target_link_libraries(test_seeded_key_material_and_query_roundtrip PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_seeded_key_material_and_query_roundtrip
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_full_scoring_with_splits_seeded tests/test_full_scoring_with_splits_seeded.cpp)
target_link_libraries(test_full_scoring_with_splits_seeded PRIVATE
        mimir_core
        GTest::gtest_main
)
gtest_discover_tests(test_full_scoring_with_splits_seeded
        DISCOVERY_MODE PRE_TEST
)

add_executable(test_full_scoring_with_splits_seeded_parallel tests/test_full_scoring_with_splits_seeded_parallel.cpp)
target_link_libraries(test_full_scoring_with_splits_seeded_parallel PRIVATE
        mimir_core
        GTest::gtest_main
        OpenMP::OpenMP_CXX
)
gtest_discover_tests(test_full_scoring_with_splits_seeded_parallel
        DISCOVERY_MODE PRE_TEST
)


# ----- Tests for the server setup and serving ------
add_executable(test_server_database_value_range tests/test_server_database_value_range.cpp)
target_link_libraries(test_server_database_value_range PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_server_database_value_range DISCOVERY_MODE PRE_TEST)

add_executable(test_real_database_loading tests/test_real_database_loading.cpp)
target_link_libraries(test_real_database_loading PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_real_database_loading DISCOVERY_MODE PRE_TEST)

add_executable(test_database_scoring_correctness tests/test_database_scoring_correctness.cpp)
target_link_libraries(test_database_scoring_correctness PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_database_scoring_correctness DISCOVERY_MODE PRE_TEST)

add_executable(test_database_scoring_correctness_with_splits tests/test_database_scoring_correctness_with_splits.cpp)
target_link_libraries(test_database_scoring_correctness_with_splits PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_database_scoring_correctness_with_splits DISCOVERY_MODE PRE_TEST)

add_executable(test_wire_protocol_roundtrip tests/test_wire_protocol_roundtrip.cpp)
target_link_libraries(test_wire_protocol_roundtrip PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_wire_protocol_roundtrip DISCOVERY_MODE PRE_TEST)

add_executable(test_process_query_correctness tests/test_process_query_correctness.cpp)
target_link_libraries(test_process_query_correctness PRIVATE mimir_core GTest::gtest_main OpenMP::OpenMP_CXX)
gtest_discover_tests(test_process_query_correctness DISCOVERY_MODE PRE_TEST)

add_executable(test_server_session tests/test_server_session.cpp)
target_link_libraries(test_server_session PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_server_session DISCOVERY_MODE PRE_TEST)

add_executable(test_server_database_uniform_cluster_sizes tests/test_database_uniform_cluster_sizes.cpp)
target_link_libraries(test_server_database_uniform_cluster_sizes PRIVATE mimir_core GTest::gtest_main)
gtest_discover_tests(test_server_database_uniform_cluster_sizes DISCOVERY_MODE PRE_TEST)


# ------ Latency Benchmarks ------
find_package(OpenMP REQUIRED)

# benchmarks with full randomness a in ciphertexts
add_executable(benchmark_latency benchmarks/benchmark_latency.cpp)
target_link_libraries(benchmark_latency PRIVATE mimir_core)

add_executable(benchmark_latency_parallel benchmarks/benchmark_latency_parallel.cpp)
target_link_libraries(benchmark_latency_parallel PRIVATE mimir_core OpenMP::OpenMP_CXX)

add_executable(benchmark_latency_distributed benchmarks/benchmark_latency_distributed.cpp)
target_link_libraries(benchmark_latency_distributed PRIVATE mimir_core OpenMP::OpenMP_CXX)

# benchmarks with seed
add_executable(benchmark_latency_seeded benchmarks/benchmark_latency_seeded.cpp)
target_link_libraries(benchmark_latency_seeded PRIVATE mimir_core)

add_executable(benchmark_latency_seeded_parallel benchmarks/benchmark_latency_seeded_parallel.cpp)
target_link_libraries(benchmark_latency_seeded_parallel PRIVATE mimir_core OpenMP::OpenMP_CXX)

add_executable(benchmark_latency_seeded_distributed benchmarks/benchmark_latency_seeded_distributed.cpp)
target_link_libraries(benchmark_latency_seeded_distributed PRIVATE mimir_core OpenMP::OpenMP_CXX)


# ----- Server and Client Application ------
add_executable(server apps/server.cpp)
target_link_libraries(server PRIVATE mimir_core OpenMP::OpenMP_CXX)