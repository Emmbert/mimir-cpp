#pragma once
#include <cstdint>
#include <string>

namespace psearch {

struct DatabaseMetadata {
    int64_t precision;
    double scale;
    std::string dtype;
    int64_t embedding_length;
    bool normalized;
};

/// Loads the .meta.json sidecar convert_clusters_to_database.py writes
/// alongside a .mdb database file (same path with ".meta.json" appended).
/// Throws std::runtime_error if the sidecar doesn't exist or can't be
/// parsed, or if any expected field is missing.
DatabaseMetadata load_database_metadata(const std::string& mdb_path);

} // namespace psearch
