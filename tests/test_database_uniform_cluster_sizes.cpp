// test_server_database_uniform_cluster_sizes.cpp
//
// Tests validate_uniform_cluster_sizes against SYNTHETIC data -- same
// reasoning as test_server_database_value_range.cpp: your real clustering
// pipeline presumably produces uniform cluster sizes by construction, so a
// test against real data would only ever exercise the happy path. This
// constructs databases with deliberately UNEVEN cluster sizes to confirm
// the check actually catches that.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_server_database_uniform_cluster_sizes

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <vector>

#include "db_polynomial.hpp"
#include "params.hpp"
#include "server_db.hpp"

using namespace psearch;

namespace {

std::filesystem::path write_synthetic_database_file(const std::vector<std::vector<int8_t>>& clusters,
                                                      uint32_t embedding_length) {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() / "test_server_database_uniform_cluster_sizes_tmp.mdb";

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

TEST(ServerDatabaseUniformClusterSizes, UniformClustersPassValidation) {
    Params params = Params::make_test_params();
    params.embedding_length = 2;
    params.database_size = 8; // 2 clusters x 4 embeddings each -- uniform
    params.num_clusters = 2;
    params.derive_dependent_parameters();

    std::vector<int8_t> cluster0(4 * static_cast<size_t>(params.embedding_length), 1);
    std::vector<int8_t> cluster1(4 * static_cast<size_t>(params.embedding_length), 1);
    auto path = write_synthetic_database_file({cluster0, cluster1}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_NO_THROW(validate_uniform_cluster_sizes(db, params));

    std::filesystem::remove(path);
}

TEST(ServerDatabaseUniformClusterSizes, UnevenClustersFailValidation) {
    Params params = Params::make_test_params();
    params.embedding_length = 2;
    params.database_size = 8;
    params.num_clusters = 2; // params.cluster_size = 4
    params.derive_dependent_parameters();

    std::vector<int8_t> cluster0(4 * static_cast<size_t>(params.embedding_length), 1); // matches
    std::vector<int8_t> cluster1(6 * static_cast<size_t>(params.embedding_length), 1); // does NOT match
    auto path = write_synthetic_database_file({cluster0, cluster1}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_THROW(validate_uniform_cluster_sizes(db, params), std::runtime_error);

    std::filesystem::remove(path);
}

TEST(ServerDatabaseUniformClusterSizes, ClustersUniformWithEachOtherButNotMatchingParamsFailValidation) {
    // Both clusters agree with EACH OTHER (5 embeddings each), but neither
    // matches params.cluster_size (4, from database_size/num_clusters) --
    // checks that validate_uniform_cluster_sizes compares against
    // params.cluster_size specifically, not just cluster-to-cluster
    // agreement (which alone wouldn't catch a stale/mismatched Params).
    Params params = Params::make_test_params();
    params.embedding_length = 2;
    params.database_size = 8;
    params.num_clusters = 2; // params.cluster_size = 4
    params.derive_dependent_parameters();

    std::vector<int8_t> cluster0(5 * static_cast<size_t>(params.embedding_length), 1);
    std::vector<int8_t> cluster1(5 * static_cast<size_t>(params.embedding_length), 1);
    auto path = write_synthetic_database_file({cluster0, cluster1}, static_cast<uint32_t>(params.embedding_length));

    ServerDatabase db = ServerDatabase::load_from_file(path.string());
    EXPECT_THROW(validate_uniform_cluster_sizes(db, params), std::runtime_error);

    std::filesystem::remove(path);
}