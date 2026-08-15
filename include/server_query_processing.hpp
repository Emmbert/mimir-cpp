#pragma once
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "server_db.hpp"
#include "wire_protocol.hpp"

namespace psearch {

/// Runs the full protocol (Steps 5-7) against the real loaded database:
/// unpacks the seeded query (sequential -- see seeded_query.hpp), then RLWE
/// switching (embedding), RGSW switching (selector), per-(cluster,split)
/// scoring, and cross-cluster summation, all parallelized with OpenMP --
/// exactly the pattern in benchmark_latency_parallel.cpp, with
/// ServerDatabase::build_split standing in for the benchmark's random
/// polynomial generation. Assumes every cluster has the same size (see
/// validate_uniform_cluster_sizes, called once at server startup, not
/// re-checked per query).
QueryResponse process_query(const CryptoContext& ctx, const Params& params, const ServerDatabase& db,
                             const ClientPublicMaterial& pub, const SeededQuery& query_wire);

} // namespace psearch
