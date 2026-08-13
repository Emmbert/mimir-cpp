// benchmark_latency_distributed.cpp
//
// Simulates a deployment with num_servers identical machines, each holding
// clusters_per_server = num_clusters / num_servers clusters, WITHOUT
// actually running num_servers processes or any real network/IPC. This is a
// COMPUTE-ONLY latency estimate: it answers "how long would the crypto work
// take if perfectly parallelized across num_servers identical machines,"
// deliberately excluding network round-trip time, serialization, and
// orchestration overhead -- those are a separate, differently-reportable
// number (see the much earlier discussion on disentangling compute latency
// from real distributed-deployment latency; this is the "compute" half).
//
// What's actually simulated vs. what's computed for real:
//   - Query encryption, the (single, shared) RLWE switch of the embedding
//     vector, and decryption: computed IN FULL, unaffected by num_servers --
//     these aren't sharded across servers in the real protocol either.
//   - RGSW switching ("unpacking" the cluster-selector bits) and scoring
//     (including the RGSW mask): computed for REAL, but only for
//     clusters_per_server clusters, not all num_clusters. Every server is
//     assumed identical and works in parallel, so this one computation's
//     wall-clock time stands in for what EVERY server experiences
//     simultaneously -- there's no reason to multiply it by num_servers.
//   - Cross-server summation: the orchestrator receiving one partial-sum
//     ciphertext per server (per split) and adding all num_servers of them
//     together. This genuinely needs to happen num_servers times, so the one
//     partial result actually computed above is added into the final
//     accumulator num_servers times, simulating "as if" num_servers
//     structurally-identical partials had arrived from num_servers real
//     machines. This is the one step whose cost actually scales with
//     num_servers in this benchmark.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_distributed

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "timing.hpp"
#include "params_io.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

constexpr int kSetupWarmupRuns = 2;
constexpr int kSetupMeasuredRuns = 10;

constexpr int kQueryWarmupRuns = 2;
constexpr int kQueryMeasuredRuns = 10;

const char* kOutputFilePath = "benchmark_latency_distributed_results.txt";

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

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

void run_setup_and_registration(const CryptoContext& ctx, const Params& params, LatencyRecorder& rec) {
    ClientSecretMaterial secret;
    {
        ScopedTimer t(rec, "client_setup");
        secret = generate_client_secret_material(ctx, params);
    }
    {
        ScopedTimer t(rec, "client_registration");
        [[maybe_unused]] ClientPublicMaterial pub = generate_client_public_material(ctx, secret);
    }
}

void run_one_query(const CryptoContext& ctx, const Params& params, const ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec) {
    if (params.num_servers <= 0 || params.clusters_per_server <= 0) {
        throw std::invalid_argument("params.num_servers and params.clusters_per_server must be > 0 "
                                     "for the distributed benchmark; check derive_dependent_parameters().");
    }

    // --- client_query_gen: sequential -- unaffected by server topology. --
    std::vector<LWECT> embedding_lwe;
    std::vector<LWEGadgetCT> selector_gadget; // only clusters_per_server of these get switched below
    {
        ScopedTimer t(rec, "client_query_gen");

        embedding_lwe.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            int64_t m = sample_signed_mod_value(params, rng);
            embedding_lwe.push_back(secret.lwe_sk->encode_and_encrypt(m, ctx.encoding));
        }

        // The client still doesn't know server topology -- it builds
        // selector bits for every cluster as usual. We only USE
        // clusters_per_server of them below, to simulate one server's share.
        selector_gadget.reserve(static_cast<size_t>(params.clusters_per_server));
        for (int64_t c = 0; c < params.clusters_per_server; ++c) {
            int64_t bit = (c == params.desired_cluster_index % params.clusters_per_server) ? 1 : 0;
            selector_gadget.push_back(secret.lwe_gadget_sk->gadget_encrypt(bit));
        }
    }

    std::vector<RLWECT> final_result;
    std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval;
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(static_cast<size_t>(params.clusters_per_server));

    {
        ScopedTimer t(rec, "server_processing");

        // --- Shared, non-sharded step: convert the embedding query to
        // RLWE/eval form once, in full -- every server needs the same query,
        // this isn't divided by num_servers. -------------------------------
        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            query_eval.resize(static_cast<size_t>(params.embedding_length));
            #pragma omp parallel for schedule(dynamic)
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, embedding_lwe[static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
            }
        }

        // --- Per-server step: RGSW switching, only clusters_per_server of
        // them -- this stands in for ONE server's latency; every server
        // does the same amount of work, in parallel, at the same time. -----
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t c = 0; c < params.clusters_per_server; ++c) {
                RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(selector_gadget[static_cast<size_t>(c)]);
                rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
            }
        }

        // --- Per-server step: scoring, only clusters_per_server clusters. -
        std::vector<RLWECT> partial_result; // one server's contribution, per split
        {
            ScopedTimer t_scoring(rec, "scoring calculations");

            partial_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                partial_result.emplace_back(ctx.rlwe_param);
            }

            std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
                static_cast<size_t>(params.splits_per_cluster));
            for (auto& row : masked) {
                row.resize(static_cast<size_t>(params.clusters_per_server));
            }

            int64_t total_pairs = params.clusters_per_server * params.splits_per_cluster;

            std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
            for (auto& r : thread_rngs) r.seed(rng());

            using Clock = std::chrono::steady_clock;
            std::chrono::duration<double, std::milli> db_build_time{0};
            std::chrono::duration<double, std::milli> score_time{0};
            std::chrono::duration<double, std::milli> mask_time{0};

            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_pairs; ++idx) {
                int64_t c = idx / params.splits_per_cluster;
                int64_t s = idx % params.splits_per_cluster;
                std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

                auto t0 = Clock::now();
                std::vector<DatabasePolynomialEvalForm> db_split;
                db_split.reserve(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    db_split.push_back(build_random_database_polynomial_eval_form(ctx, params, local_rng));
                }
                auto t1 = Clock::now();

                RLWECT score = compute_split_score(ctx, query_eval, db_split);
                auto t2 = Clock::now();

                RLWECT masked_val(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);
                auto t3 = Clock::now();

                masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);

                #pragma omp critical
                {
                    db_build_time += (t1 - t0);
                    score_time += (t2 - t1);
                    mask_time += (t3 - t2);
                }
            }

            rec.add_sample("  -> db polynomial building (one server)", db_build_time.count());
            rec.add_sample("  -> score computation (one server)", score_time.count());
            rec.add_sample("  -> RGSW masking (one server)", mask_time.count());

            // Sequential reduction over THIS server's own clusters only.
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t c = 0; c < params.clusters_per_server; ++c) {
                    partial_result[static_cast<size_t>(s)].add(
                        partial_result[static_cast<size_t>(s)],
                        *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
                }
            }
        }

        // --- Cross-server summation: the orchestrator combining one
        // partial result per server. Genuinely repeated num_servers times --
        // this is the one step whose cost actually scales with num_servers
        // here, since we're simulating num_servers arrivals by reusing the
        // single real partial_result computed above. -----------------------
        {
            ScopedTimer t_cross_server(rec, "cross-server summation");

            final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result.emplace_back(ctx.rlwe_param);
            }

            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t server = 0; server < params.num_servers; ++server) {
                    final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)],
                                                               partial_result[static_cast<size_t>(s)]);
                }
            }
        }
    }

    {
        ScopedTimer t(rec, "client_decrypt");
        for (const auto& ct : final_result) {
            [[maybe_unused]] Vector decrypted = secret.rlwe_sk->decrypt_vector(ct, ctx.encoding);
        }
    }
}

} // namespace


int main(int argc, char** argv) {
    Params params = load_benchmark_params_from_args(argc, argv);
    CryptoContext ctx = CryptoContext::from_params(params);

    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
    std::cout << "num_servers: " << params.num_servers << ", clusters_per_server: "
              << params.clusters_per_server << " (of " << params.num_clusters << " total clusters)\n\n";

    std::mt19937_64 rng(std::random_device{}());

    LatencyRecorder rec;

    std::cout << "Warming up client setup/registration (" << kSetupWarmupRuns << " runs)...\n";
    for (int i = 0; i < kSetupWarmupRuns; ++i) run_setup_and_registration(ctx, params, rec);
    rec.clear();

    std::cout << "Measuring client setup/registration (" << kSetupMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kSetupMeasuredRuns; ++i) run_setup_and_registration(ctx, params, rec);

    std::cout << "\n=== Client setup / registration latency ===\n";
    rec.print_summary();

    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

    LatencyRecorder query_rec;

    std::cout << "\nWarming up per-query pipeline (" << kQueryWarmupRuns << " runs)...\n";
    for (int i = 0; i < kQueryWarmupRuns; ++i) run_one_query(ctx, params, secret, pub, rng, query_rec);
    query_rec.clear();

    std::cout << "Measuring per-query pipeline (" << kQueryMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kQueryMeasuredRuns; ++i) run_one_query(ctx, params, secret, pub, rng, query_rec);

    std::cout << "\n=== Per-query latency (simulated " << params.num_servers << "-server deployment, "
              << "compute-only, no network cost) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        out << "num_servers: " << params.num_servers << ", clusters_per_server: "
            << params.clusters_per_server << " (of " << params.num_clusters << " total clusters)\n";
        out << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
        out << "=== Client setup / registration latency ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (compute-only, no network cost) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}
