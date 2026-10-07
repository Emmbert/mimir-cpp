#pragma once
#include <cstdint>
#include <ostream>
#include <vector>

#include "params.hpp"

namespace psearch {

/// Communication cost of one client, in bytes (double, because the
/// theoretical numbers are bit counts divided by 8).
struct CommunicationCost {
    double keys_bytes = 0.0;     ///< one-time eval keys (registration)
    double upload_bytes = 0.0;   ///< per query, client -> server
    double download_bytes = 0.0; ///< per query, server -> client

    double per_query_bytes() const { return upload_bytes + download_bytes; }
};

/// Theoretical cost, counting 128 bits per seed and ceil(log2 q) bits per
/// transmitted coefficient, with no byte alignment:
///
///   keys     = 128 + (log2(n) * d_ksk * n + 2 * d' * n) * ceil(log2 q)
///   upload   = r * 128 + (r * l + C * d') * ceil(log2 q)
///   download = r * s * 2 * n * ceil(log2 q)
///
/// Digit counts are taken from the loaded Params (d = ceil(log_B q)), i.e.
/// exactly what the code builds.
CommunicationCost theoretical_communication_cost(const Params& params);

/// Sizes of the byte streams that were actually produced.
struct MeasuredStreamSizes {
    std::size_t keys_bytes = 0;
    std::vector<std::size_t> upload_stream_bytes;   ///< [ring]
    std::vector<std::size_t> download_stream_bytes; ///< [ring]
    bool consistent = true; ///< false if sizes ever differed between runs

    /// Records one run's sizes; flags `consistent = false` on any change.
    void record_keys(std::size_t bytes);
    void record_query(const std::vector<std::size_t>& upload, const std::vector<std::size_t>& download);

    CommunicationCost as_cost() const;
};

/// Prints the "=== Communication cost ===" block: one-time keys, upload,
/// download and total per query (upload + download), theoretical vs actual,
/// in bytes and MiB, plus the per-ring stream breakdown.
void print_communication_cost(std::ostream& os, const Params& params, const CommunicationCost& theoretical,
                              const MeasuredStreamSizes& actual);

} // namespace psearch
