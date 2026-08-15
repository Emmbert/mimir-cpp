#include "server_db.hpp"

#include <fstream>
#include <limits>
#include <stdexcept>

namespace psearch {

namespace {

uint32_t read_u32(std::istream& in, const std::string& path) {
    uint32_t v;
    in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) {
        throw std::runtime_error("ServerDatabase::load_from_file: " + path +
                                  ": unexpected end of file while reading header");
    }
    return v;
}

} // namespace

ServerDatabase ServerDatabase::load_from_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ServerDatabase::load_from_file: could not open " + path);
    }

    uint32_t magic = read_u32(in, path);
    if (magic != kServerDatabaseFileMagic) {
        throw std::runtime_error("ServerDatabase::load_from_file: " + path +
                                  " does not start with the expected 'MDB1' magic -- wrong file or format?");
    }

    ServerDatabase db;
    uint32_t num_clusters = read_u32(in, path);
    db.embedding_length_ = read_u32(in, path);

    db.cluster_sizes_.resize(num_clusters);
    for (uint32_t c = 0; c < num_clusters; ++c) {
        db.cluster_sizes_[c] = read_u32(in, path);
    }

    db.cluster_data_.resize(num_clusters);
    for (uint32_t c = 0; c < num_clusters; ++c) {
        size_t bytes_needed = static_cast<size_t>(db.cluster_sizes_[c]) * db.embedding_length_;
        db.cluster_data_[c].resize(bytes_needed);
        in.read(reinterpret_cast<char*>(db.cluster_data_[c].data()), static_cast<std::streamsize>(bytes_needed));
        if (!in) {
            throw std::runtime_error("ServerDatabase::load_from_file: " + path +
                                      " truncated -- expected more data for cluster " + std::to_string(c));
        }
    }

    return db;
}

int64_t ServerDatabase::splits_in_cluster(int64_t cluster, const Params& params) const {
    int64_t size = cluster_size(cluster);
    return (size + params.n - 1) / params.n; // ceil division
}

std::vector<DatabasePolynomialEvalForm> ServerDatabase::build_split(const CryptoContext& ctx, const Params& params,
                                                                      int64_t cluster, int64_t split) const {
    if (static_cast<int64_t>(embedding_length_) != params.embedding_length) {
        throw std::invalid_argument("ServerDatabase::build_split: database's embedding_length (" +
                                     std::to_string(embedding_length_) + ") does not match params.embedding_length (" +
                                     std::to_string(params.embedding_length) + ")");
    }

    const std::vector<int8_t>& data = cluster_data_.at(static_cast<size_t>(cluster));
    int64_t size = cluster_size(cluster);
    int64_t split_start = split * params.n; // first embedding index covered by this split

    std::vector<DatabasePolynomialEvalForm> db_split;
    db_split.reserve(static_cast<size_t>(params.embedding_length));

    for (int64_t j = 0; j < params.embedding_length; ++j) {
        // Zero-initialized: embeddings past this cluster's actual size
        // (only possible in the last, partial split) stay 0 -- contributes
        // nothing to the dot product.
        std::vector<int64_t> raw_values(static_cast<size_t>(params.n), 0);

        for (int64_t i = 0; i < params.n; ++i) {
            int64_t embedding_idx = split_start + i;
            if (embedding_idx < size) {
                // Embedding-major layout: data[embedding_idx * embedding_length + j].
                // int8_t -> int64_t sign-extends correctly (e.g. -7 stays -7).
                raw_values[static_cast<size_t>(i)] = static_cast<int64_t>(
                    data[static_cast<size_t>(embedding_idx) * embedding_length_ + static_cast<size_t>(j)]);
            }
        }

        db_split.push_back(build_database_polynomial_eval_form_from_raw_values(ctx, params, raw_values));
    }

    return db_split;
}

std::pair<int64_t, int64_t> ServerDatabase::scan_value_range() const {
    int64_t min_val = std::numeric_limits<int64_t>::max();
    int64_t max_val = std::numeric_limits<int64_t>::min();
    for (const auto& cluster : cluster_data_) {
        for (int8_t v : cluster) {
            int64_t value = static_cast<int64_t>(v);
            if (value < min_val) min_val = value;
            if (value > max_val) max_val = value;
        }
    }
    return {min_val, max_val};
}

void validate_value_range(const ServerDatabase& db, const Params& params) {
    auto [min_val, max_val] = db.scan_value_range();
    int64_t allowed_min = min_embedding_value(params);
    int64_t allowed_max = max_embedding_value(params);
    if (min_val < allowed_min || max_val > allowed_max) {
        throw std::runtime_error(
            "validate_value_range: loaded database contains values in [" + std::to_string(min_val) + ", " +
            std::to_string(max_val) + "], but params.embedding_precision (" +
            std::to_string(params.embedding_precision) + ") only allows [" + std::to_string(allowed_min) + ", " +
            std::to_string(allowed_max) +
            "]. Either embedding_precision is set too low for this data, or the data isn't quantized the way "
            "you expect.");
    }
}

void validate_uniform_cluster_sizes(const ServerDatabase& db, const Params& params) {
    for (int64_t c = 0; c < db.num_clusters(); ++c) {
        if (db.cluster_size(c) != params.cluster_size) {
            throw std::runtime_error(
                "validate_uniform_cluster_sizes: cluster " + std::to_string(c) + " has size " +
                std::to_string(db.cluster_size(c)) + ", but params.cluster_size (database_size / num_clusters) is " +
                std::to_string(params.cluster_size) +
                ". Query processing assumes every cluster has the same size -- if that's no longer true for "
                "this database, splits_per_cluster needs to be handled per-cluster, not uniformly.");
        }
    }
}

} // namespace psearch