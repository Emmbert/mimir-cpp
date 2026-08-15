#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "db_polynomial.hpp"
#include "params.hpp"

namespace psearch {

/// On-disk magic number for the combined database file format produced by
/// convert_clusters_to_database.py. Exposed here (not just inside
/// server_db.cpp) so anything that needs to construct a valid file --
/// tooling, tests -- shares this one definition rather than duplicating the
/// literal value.
constexpr uint32_t kServerDatabaseFileMagic = 0x4D444231; // 'MDB1'

/// Loads the combined binary database file produced by
/// convert_clusters_to_database.py: magic ('MDB1'), num_clusters,
/// embedding_length, cluster_sizes[], then each cluster's raw int8
/// embeddings (embedding-major, concatenated in cluster order). See that
/// script's header comment for the exact byte layout.
///
/// Real clusters can have different sizes from each other -- unlike
/// Params::make_test_params()/make_benchmark_params(), which assume a
/// single uniform database_size split evenly across num_clusters, this
/// class tracks each cluster's actual size and computes splits_per_cluster
/// PER CLUSTER accordingly.
class ServerDatabase {
public:
    static ServerDatabase load_from_file(const std::string& path);

    int64_t num_clusters() const { return static_cast<int64_t>(cluster_sizes_.size()); }
    int64_t embedding_length() const { return static_cast<int64_t>(embedding_length_); }
    int64_t cluster_size(int64_t cluster) const { return cluster_sizes_.at(static_cast<size_t>(cluster)); }

    /// ceil(cluster_size(cluster) / params.n).
    int64_t splits_in_cluster(int64_t cluster, const Params& params) const;

    /// Builds the embedding_length() DatabasePolynomialEvalForm objects for
    /// (cluster, split) -- same shape/semantics as
    /// build_random_database_polynomial_eval_form's per-split output
    /// (see db_polynomial.hpp), just from real loaded data instead of
    /// random sampling; both go through the same underlying
    /// build_database_polynomial_eval_form_from_raw_values, so they can't
    /// silently diverge. Embeddings beyond this cluster's actual size
    /// (only ever in the last, partial split) are zero-padded.
    ///
    /// Recomputes (raw bytes -> reduced -> NTT) fresh on every call,
    /// matching how build_random_database_polynomial_eval_form is already
    /// called fresh per query in every existing test/benchmark. Caching the
    /// NTT'd form persistently across queries is a natural future
    /// optimization once real per-query latency numbers are in hand for
    /// the real database -- not built yet, since it adds real complexity
    /// (thread-safety under the multithreaded server, memory sizing for a
    /// full eval-form cache) that deserves to be a deliberate decision, not
    /// a default.
    std::vector<DatabasePolynomialEvalForm> build_split(const CryptoContext& ctx, const Params& params,
                                                          int64_t cluster, int64_t split) const;

    /// Scans every loaded embedding value and returns {min, max}. Useful as
    /// a one-time sanity check that params.embedding_precision's range
    /// actually covers the real data -- if it doesn't, dot_product_can_overflow's
    /// guarantee (which is based on embedding_precision alone, not the
    /// actual data) would be silently wrong for this database. Not called
    /// automatically anywhere; call it explicitly after loading if you want
    /// the check.
    std::pair<int64_t, int64_t> scan_value_range() const;

private:
    uint32_t embedding_length_ = 0;
    std::vector<uint32_t> cluster_sizes_;
    std::vector<std::vector<int8_t>> cluster_data_; // [cluster] -> raw embedding-major bytes
};

/// Checks that db.scan_value_range() falls within
/// [min_embedding_value(params), max_embedding_value(params)]. Throws
/// std::runtime_error with a clear message if not -- meant to be called
/// once, right after loading a real database, so a mismatched
/// embedding_precision fails loudly immediately rather than silently
/// producing overflow-unsafe results at query time. dot_product_can_overflow
/// (db_polynomial.hpp) can't catch this on its own -- it's based on
/// embedding_precision alone, never on the actual loaded data.
void validate_value_range(const ServerDatabase& db, const Params& params);

/// Checks that every cluster in db has EXACTLY the same size, and that it
/// equals params.cluster_size (database_size / num_clusters). Throws
/// std::runtime_error with a clear message if not. Query processing assumes
/// a uniform splits_per_cluster across every cluster (matching every
/// benchmark's assumption) -- if that ever stops holding for some future
/// dataset, this is what catches it at startup instead of silently
/// producing wrong results (a cluster with fewer splits than params.n
/// assumes would either read out of bounds or silently miss real data).
void validate_uniform_cluster_sizes(const ServerDatabase& db, const Params& params);

} // namespace psearch