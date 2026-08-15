// test_server_database_value_range.cpp
//
// Tests validate_value_range against SYNTHETIC data with known values,
// deliberately placed at and past the boundary -- real quantized embedding
// data presumably never violates its own precision, so a test against real
// data would only ever exercise the happy path. Constructing the synthetic
// file directly (rather than loading a real one) is what actually lets this
// test cover the failure case: a mismatched embedding_precision setting
// silently letting through data with a wider range than the parameters
// assume.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_server_database_value_range

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "db_polynomial.hpp"
#include "params.hpp"
#include "server_db.hpp"

using namespace psearch;

namespace {

/// Writes a minimal, valid on-disk database file (see server_db.hpp /
/// convert_clusters_to_database.py for the format) directly from in-memory
/// per-cluster data, so tests don't depend on any external fixture file.
std::filesystem::path write_synthetic_database_file(const std::vector<std::vector<int8_t>>& clusters,
                                                      uint32_t embedding_length) {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / "test_server_database_value_range_tmp.mdb";

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

} // namespace

TEST(ServerDatabaseValueRange, InRangeDataPassesValidation) {
    Params params = Params::make_test_params();
    params.embedding_precision = 6; // signed range [-32, 31], matching the real deployment's data
    params.embedding_length = 4;    // small, only what this test needs

    // Every value strictly within [-32, 31], including both boundary values.
    std::vector<int8_t> cluster0 = {-32, 31, 0, 5, -32, 31, 0, -5};
    auto path = write_synthetic_database_file({cluster0}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_NO_THROW(validate_value_range(db, params));

    std::filesystem::remove(path);
}

TEST(ServerDatabaseValueRange, OutOfRangeDataFailsValidation) {
    Params params = Params::make_test_params();
    params.embedding_precision = 6; // signed range [-32, 31]
    params.embedding_length = 4;

    // One value (32) is one past the allowed maximum (31) -- everything
    // else is comfortably in range, so this specifically tests that a
    // single violating value anywhere in the database is caught, not just
    // gross out-of-range data throughout.
    std::vector<int8_t> cluster0 = {-32, 31, 0, 5, -10, 32, 0, -5};
    auto path = write_synthetic_database_file({cluster0}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_THROW(validate_value_range(db, params), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(ServerDatabaseValueRange, OutOfRangeBelowMinimumFailsValidation) {
    Params params = Params::make_test_params();
    params.embedding_precision = 6; // signed range [-32, 31]
    params.embedding_length = 4;

    // -33 is one past the allowed minimum (-32).
    std::vector<int8_t> cluster0 = {-32, 31, 0, 5, -33, 10, 0, -5};
    auto path = write_synthetic_database_file({cluster0}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_THROW(validate_value_range(db, params), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(ServerDatabaseValueRange, MultipleClustersAllChecked) {
    Params params = Params::make_test_params();
    params.embedding_precision = 6;
    params.embedding_length = 4;

    // Violation is in the SECOND cluster, not the first -- checks that
    // scan_value_range actually scans every cluster, not just the first.
    std::vector<int8_t> cluster0 = {-32, 31, 0, 5};
    std::vector<int8_t> cluster1 = {-10, 10, 0, 40}; // 40 violates the max (31)
    auto path = write_synthetic_database_file({cluster0, cluster1}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_THROW(validate_value_range(db, params), std::runtime_error);

    std::filesystem::remove(path);
}
