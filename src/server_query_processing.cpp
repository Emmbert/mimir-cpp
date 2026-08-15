#include "server_query_processing.hpp"

#include <omp.h>

#include <memory>
#include <vector>

using namespace FHEDeck;

namespace psearch {

namespace {

RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<std::unique_ptr<RLWECTEvalForm>>& query_eval,
                            const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j]->mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

} // namespace

QueryResponse process_query(const CryptoContext& ctx, const Params& params, const ServerDatabase& db,
                             const ClientPublicMaterial& pub, const SeededQuery& query_wire) {
    // --- Unpacking: sequential -- see the earlier discussion on why
    // SeededUniformDistribution's rejection sampling makes this
    // unparallelizable as currently implemented. ------------------------------
    ReconstructedQuery query = reconstruct_query(ctx, params, query_wire);

    // --- RLWE switching: full embedding, parallel. ----------------------------
    std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval(query.embedding_cts.size());
    #pragma omp parallel for schedule(dynamic)
    for (int64_t j = 0; j < static_cast<int64_t>(query.embedding_cts.size()); ++j) {
        RLWECT rlwe_ct(ctx.rlwe_param);
        pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, query.embedding_cts[static_cast<size_t>(j)]);
        query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
    }

    // --- RGSW switching: full selector set, parallel. ---------------------------
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(query.selector_cts.size());
    #pragma omp parallel for schedule(dynamic)
    for (int64_t c = 0; c < static_cast<int64_t>(query.selector_cts.size()); ++c) {
        RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]);
        rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
    }

    // --- Per-(cluster,split) scoring: real data via db.build_split, in
    // place of the benchmark's random polynomial generation. Every cluster
    // has the same size (validated once at startup), so
    // params.splits_per_cluster applies uniformly -- same structure as
    // benchmark_latency_parallel.cpp. --------------------------------------------
    std::vector<RLWECT> final_result;
    final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
    for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
        final_result.emplace_back(ctx.rlwe_param);
    }

    std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(static_cast<size_t>(params.splits_per_cluster));
    for (auto& row : masked) {
        row.resize(static_cast<size_t>(params.num_clusters));
    }

    int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

    #pragma omp parallel for schedule(dynamic)
    for (int64_t idx = 0; idx < total_pairs; ++idx) {
        int64_t c = idx / params.splits_per_cluster;
        int64_t s = idx % params.splits_per_cluster;

        std::vector<DatabasePolynomialEvalForm> db_split = db.build_split(ctx, params, c, s);

        RLWECT score = compute_split_score(ctx, query_eval, db_split);

        RLWECT masked_val(ctx.rlwe_param);
        rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);

        masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);
    }

    // --- Sequential reduction: sum over clusters, per split. --------------------
    for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)],
                                                       *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
        }
    }

    QueryResponse response;
    response.ciphertexts = std::move(final_result);
    return response;
}

} // namespace psearch
