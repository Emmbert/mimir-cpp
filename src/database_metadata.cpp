#include "database_metadata.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace psearch {

DatabaseMetadata load_database_metadata(const std::string& mdb_path) {
    std::string sidecar_path = mdb_path + ".meta.json";
    std::ifstream in(sidecar_path);
    if (!in) {
        throw std::runtime_error("load_database_metadata: could not open " + sidecar_path);
    }

    nlohmann::json j;
    try {
        in >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("load_database_metadata: malformed JSON in " + sidecar_path + ": " + e.what());
    }

    DatabaseMetadata meta;
    meta.precision = j.at("precision").get<int64_t>();
    meta.scale = j.at("scale").get<double>();
    meta.dtype = j.at("dtype").get<std::string>();
    meta.embedding_length = j.at("embedding_length").get<int64_t>();
    meta.normalized = j.at("normalized").get<bool>();
    return meta;
}

} // namespace psearch
