#include "server.hpp"

#include <iostream>

namespace psearch {

Server::Server(const Params& params, std::shared_ptr<const CryptoContext> ctx, ServerDatabase db,
               std::unique_ptr<IWorkerChannel> channel)
    : params_(params), ctx_(std::move(ctx)), db_(std::move(db)), channel_(std::move(channel)) {}

void Server::register_client(const std::string& client_id, ClientEvalKeys keys) {
    // Step 3: "unpack the LWE cts into RLWE ciphertexts and store them" per the
    // original protocol description maps, in FHE-Deck terms, to storing the
    // key-switching keys the client generated for us. TODO: confirm whether
    // FHEDeck::LWEToRLWEKeySwitchKey / LWEToRGSWKeySwitchKey need any explicit
    // "expand from seed" step server-side (as crypto.py's
    // expand_KS_public_key_from_seed does) or are already fully usable as
    // received. If FHE-Deck ships them "compressed" with a seed, the expansion
    // happens here.
    sessions_[client_id] = ClientSession{std::move(keys.lwe_to_rlwe_ksk), std::move(keys.lwe_to_rgsw_ksk)};
}

std::vector<FHEDeck::RLWECTEvalForm> Server::switch_embedding_to_rlwe(
    const ClientSession& session, const std::vector<FHEDeck::LWECT>& embedding_cts) const {
    std::vector<FHEDeck::RLWECTEvalForm> result;
    result.reserve(embedding_cts.size());
    for (const auto& ct : embedding_cts) {
        // TODO:
        //   FHEDeck::RLWECT rlwe_ct(ctx_->rlwe_param);
        //   session.lwe_to_rlwe_ksk.lwe_to_rlwe_key_switch(rlwe_ct, ct);
        //   result.emplace_back(rlwe_ct); // -> eval/NTT form
    }
    return result;
}

std::vector<FHEDeck::RLWEGadgetCT> Server::switch_unit_vector_to_rgsw(
    const ClientSession& session, const std::vector<FHEDeck::LWECT>& unit_vector_cts) const {
    std::vector<FHEDeck::RLWEGadgetCT> result;
    result.reserve(unit_vector_cts.size());
    for (const auto& ct : unit_vector_cts) {
        // TODO: result.push_back(session.lwe_to_rgsw_ksk.lwe_to_rlwe_key_switch(ct));
        // (NTT/eval conversion: confirm whether RLWEGadgetCT needs an explicit
        // eval-form step analogous to RLWECTEvalForm, or handles this internally.)
    }
    return result;
}

EncryptedResponse Server::process_query(const std::string& client_id, const EncryptedQuery& query) {
    const auto& session = sessions_.at(client_id);

    // --- Step 5: LWE -> RLWE / RGSW switch + NTT, then dispatch to workers ---
    auto query_rlwe = switch_embedding_to_rlwe(session, query.embedding_cts);       // length l
    auto cluster_rgsw_all = switch_unit_vector_to_rgsw(session, query.unit_vector_cts); // length c

    std::vector<std::future<WorkerResult>> pending;
    pending.reserve(static_cast<size_t>(params_.num_servers));

    for (int64_t w = 0; w < params_.num_servers; ++w) {
        WorkerJob job;
        job.query_rlwe = query_rlwe; // TODO: consider sharing via pointer/span instead of copying,
                                     // once you care about the copy cost in the benchmark.

        // this worker's slice of the c RGSW cluster-selector ciphertexts:
        int64_t start = w * params_.clusters_per_server;
        job.cluster_rgsw.assign(cluster_rgsw_all.begin() + start,
                                 cluster_rgsw_all.begin() + start + params_.clusters_per_server);

        pending.push_back(channel_->submit(static_cast<int>(w), std::move(job)));
    }

    // --- Step 6 happens inside each worker (see worker_compute) --------------

    // --- Step 7: gather + "simulate num_servers many real servers" + combine --
    std::vector<WorkerResult> per_worker_results;
    per_worker_results.reserve(pending.size());
    for (auto& f : pending) per_worker_results.push_back(f.get());

    return combine_worker_results(per_worker_results);
}

WorkerResult Server::worker_compute(const CryptoContext& ctx,
                                     const std::vector<ServerDatabase::Cluster>& worker_clusters,
                                     const WorkerJob& job) {
    WorkerResult result;
    result.resize(worker_clusters.size());

    // TODO: parallelize this outer loop with OpenMP (#pragma omp parallel for)
    // once correctness is established — clusters are fully independent.
    for (size_t c = 0; c < worker_clusters.size(); ++c) {
        const auto& cluster = worker_clusters[c]; // [split][embedding_dim]
        result[c].reserve(cluster.size());

        for (const auto& split_polys : cluster) { // split_polys: [embedding_dim] PolynomialEvalForm
            // Step 6a: sum_i( query_rlwe[i] * split_polys[i] ) -> 1 RLWE ct for this split.
            // TODO:
            //   FHEDeck::RLWECTEvalForm acc(ctx.rlwe_param);
            //   for (size_t i = 0; i < split_polys.size(); ++i) {
            //       FHEDeck::RLWECTEvalForm product(ctx.rlwe_param);
            //       job.query_rlwe[i].mul(product, split_polys[i]);
            //       acc.add(acc, product);
            //   }

            // Step 6b: multiply that split's RLWE by this cluster's RGSW selector.
            // TODO:
            //   FHEDeck::RLWECT masked(ctx.rlwe_param);
            //   job.cluster_rgsw[c].mul(masked, RLWECT(acc));
            //   result[c].push_back(RLWECTEvalForm(masked));
        }
    }
    return result;
}

EncryptedResponse Server::combine_worker_results(const std::vector<WorkerResult>& per_worker_results) const {
    // Step 7: "we simulate having many machines, but the main process only
    // really got back one worker's clusters_per_server*s RLWE cts, and we just
    // reuse this value num_servers times." Implemented here as: for each split
    // index i, sum across every (worker, local_cluster) pair *and* repeat that
    // sum num_servers times in total, per the description.
    //
    // TODO: implement precisely per Step 7's wording once you're sure whether
    // "take this value number_of_servers often" means
    //   (a) literally reuse worker 0's result object num_servers times (pure
    //       simulation artifact, only 1 worker is ever actually computed), or
    //   (b) each worker computes independently (as this scaffold currently
    //       does, calling channel_->submit for every w) and this step is just
    //       "sum across all of them".
    // Given num_servers workers are already dispatched to individually above,
    // (b) is what's implemented in process_query(); this function's job is
    // "just" the cross-cluster sum per split index, then NTT->coefficient form.

    EncryptedResponse response;
    response.resize(static_cast<size_t>(params_.splits_per_cluster));

    for (int64_t split = 0; split < params_.splits_per_cluster; ++split) {
        // TODO:
        //   FHEDeck::RLWECTEvalForm sum(ctx_->rlwe_param);
        //   for (const auto& worker_result : per_worker_results)
        //       for (const auto& cluster_result : worker_result)
        //           sum.add(sum, cluster_result[split]);
        //   response[split] = RLWECT(sum); // convert back from NTT/eval form
    }
    return response;
}

} // namespace psearch
