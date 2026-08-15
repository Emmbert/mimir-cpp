// test_real_database_loading.cpp
//
// Verifies a REAL .mdb database file (produced by convert_clusters_to_database.py)
// loads correctly, AND -- this is the part that actually matters -- that the
// values ServerDatabase::build_split extracts genuinely match the file's
// raw bytes, not just that loading doesn't crash. Does this by reading a
// handful of (cluster, embedding, dim) positions directly from the file's
// bytes, completely bypassing ServerDatabase, and comparing against what
// build_split reports for the same positions.
//
// File resolution: MIMIR_TEST_DATABASE_FILE, if set, overrides everything
// below -- useful for pointing this test at a different file without
// touching any code. Otherwise, defaults to Params::kTestDatabaseFilePath
// (the fixed, checked-in-adjacent test database) if it exists, so this
// test actually runs by default rather than needing an env var remembered
// every time. Only skips if NEITHER is available.
//
// Run:
//   ./test_real_database_loading                                    (uses the default file, if present)
//   MIMIR_TEST_DATABASE_FILE=/path/to/other.mdb ./test_real_database_loading
//   MIMIR_TEST_DATABASE_FILE=/path/to/other.mdb ctest -R RealDatabaseLoading

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "database_metadata.hpp"
#include "db_polynomial.hpp"
#include "params.hpp"
#include "server_db.hpp"

using namespace psearch;

namespace {

/// Resolves which database file to test against: MIMIR_TEST_DATABASE_FILE
/// if set, else Params::kTestDatabaseFilePath if it exists, else empty
/// (meaning: nothing to test against, caller should skip).
std::string resolve_database_path() {
    const char* env_path = std::getenv("MIMIR_TEST_DATABASE_FILE");
    if (env_path != nullptr && std::string(env_path) != "") {
        return std::string(env_path);
    }
    if (std::filesystem::exists(Params::kTestDatabaseFilePath)) {
        return Params::kTestDatabaseFilePath;
    }
    return "";
}

/// Independently reads ONE raw embedding value directly from the .mdb
/// file's bytes, bypassing ServerDatabase entirely -- this is the ground
/// truth build_split's output gets compared against. Mirrors the exact
/// on-disk layout documented in server_db.hpp / convert_clusters_to_database.py:
/// magic, num_clusters, embedding_length, cluster_sizes[], then each
/// cluster's raw bytes concatenated in order.
int8_t read_raw_value_directly(const std::string& path, int64_t target_cluster, int64_t embedding_idx,
                                int64_t dim_idx) {
    std::ifstream in(path, std::ios::binary);
    uint32_t magic = 0, num_clusters = 0, embedding_length = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&num_clusters), sizeof(num_clusters));
    in.read(reinterpret_cast<char*>(&embedding_length), sizeof(embedding_length));

    std::vector<uint32_t> cluster_sizes(num_clusters);
    in.read(reinterpret_cast<char*>(cluster_sizes.data()),
            static_cast<std::streamsize>(4 * num_clusters));

    // Skip past every cluster before target_cluster.
    for (int64_t c = 0; c < target_cluster; ++c) {
        in.seekg(static_cast<std::streamoff>(cluster_sizes[static_cast<size_t>(c)]) * embedding_length,
                  std::ios::cur);
    }
    // Seek to the specific (embedding, dim) position within target_cluster.
    in.seekg(static_cast<std::streamoff>(embedding_idx) * embedding_length + dim_idx, std::ios::cur);

    int8_t value = 0;
    in.read(reinterpret_cast<char*>(&value), 1);
    return value;
}

} // namespace

TEST(RealDatabaseLoading, LoadsAndMatchesRawFileBytes) {
    std::string path = resolve_database_path();
    if (path.empty()) {
        GTEST_SKIP() << "No database file found -- place one at " << Params::kTestDatabaseFilePath
                     << ", or set MIMIR_TEST_DATABASE_FILE to point at a different one.";
    }
    std::cout << "Testing against: " << path << "\n";

    ServerDatabase db = ServerDatabase::load_from_file(path);
    std::cout << "Loaded " << path << ": " << db.num_clusters() << " clusters, embedding_length="
              << db.embedding_length() << "\n";
    for (int64_t c = 0; c < db.num_clusters(); ++c) {
        std::cout << "  cluster " << c << ": " << db.cluster_size(c) << " embeddings\n";
    }

    DatabaseMetadata meta = load_database_metadata(path);
    std::cout << "Sidecar metadata: precision=" << meta.precision << " embedding_length=" << meta.embedding_length
              << " scale=" << meta.scale << " dtype=" << meta.dtype << " normalized=" << meta.normalized << "\n";

    ASSERT_EQ(meta.embedding_length, db.embedding_length())
        << "Sidecar's embedding_length doesn't match the .mdb file's own header -- mismatched/stale files?";

    // Small n -- fast, and sufficient here: this test checks that loaded
    // DATA is correct, not the full crypto pipeline, and raw_values (what
    // gets checked below) never touches n or q at all.
    Params params = Params::make_test_params();
    params.embedding_length = meta.embedding_length;
    params.embedding_precision = meta.precision;
    params.derive_dependent_parameters();

    EXPECT_NO_THROW(validate_value_range(db, params))
        << "Loaded database's actual value range exceeds what embedding_precision=" << meta.precision
        << " (from the sidecar) allows.";

    CryptoContext ctx = CryptoContext::from_params(params);

    // --- The actual correctness check: build_split's raw_values must
    // exactly match ground truth read directly from the file's bytes, for
    // cluster 0, split 0. ---------------------------------------------------
    ASSERT_GT(db.num_clusters(), 0);
    int64_t cluster = 0;
    int64_t split = 0;

    std::vector<DatabasePolynomialEvalForm> db_split = db.build_split(ctx, params, cluster, split);
    ASSERT_EQ(static_cast<int64_t>(db_split.size()), params.embedding_length);

    int64_t embeddings_to_check = std::min<int64_t>(db.cluster_size(cluster), params.n);
    embeddings_to_check = std::min<int64_t>(embeddings_to_check, 20); // keep the test fast

    int64_t mismatches = 0;
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        for (int64_t i = 0; i < embeddings_to_check; ++i) {
            int64_t expected = static_cast<int64_t>(read_raw_value_directly(path, cluster, i, j));
            int64_t actual = db_split[static_cast<size_t>(j)].raw_values[static_cast<size_t>(i)];
            if (actual != expected) {
                ++mismatches;
            }
            EXPECT_EQ(actual, expected) << "Mismatch at cluster " << cluster << ", embedding " << i << ", dim "
                                         << j << ": file says " << expected << ", build_split says " << actual;
        }
    }

    std::cout << "Checked " << (params.embedding_length * embeddings_to_check) << " values, " << mismatches
              << " mismatches.\n";
}