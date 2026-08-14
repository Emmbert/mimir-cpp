// benchmark_latency_seeded_distributed.cpp
//
// Combines benchmark_latency_distributed.cpp's num_servers simulation
// (compute-only, no network cost) with seeded eval keys and a seeded query,
// both reconstructed with NO secret key involved in reconstruction.
//
// IMPORTANT correction from an earlier version: the client always builds
// and sends the FULL query -- all num_clusters selector ciphertexts, not
// just clusters_per_server -- since a real client doesn't know server
// topology. Unpacking (reconstruct_query) therefore reconstructs the FULL
// query too, and this cost does NOT scale down with num_servers: because
// SeededUniformDistribution's stream is sequential (rejection sampling
// makes per-ciphertext offsets data-dependent -- see the earlier
// discussion), whoever unpacks the query has to expand the ENTIRE stream
// regardless of how many physical machines exist downstream. Only AFTER
// full reconstruction does "distribution" happen: one contiguous batch of
// clusters_per_server selector ciphertexts is sliced out of the fully-
// reconstructed set (no extra work, just indexing), and only THAT batch
// gets RGSW-switched and scored -- simulating one server's share.
//
// Unpacking stays SEQUENTIAL for the same reason as
// benchmark_latency_seeded_parallel.cpp. Switching and scoring are
// parallelized, and only over the one batch, exactly as before.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_seeded_distributed
//   OMP_NUM_THREADS=16 ./benchmark_latency_seeded_distributed params.json 8

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
#include "params_io.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"
#include "timing.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

constexpr int kSetupWarmupRuns = 2;
constexpr int kSetupMeasuredRuns = 10;

constexpr int kQueryWarmupRuns = 2;
constexpr int kQueryMeasuredRuns = 10;

const char* kOutputFilePath = "benchmark_latency_seeded_distributed_results.txt";

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

ClientPublicMaterial run_setup_and_registration(const CryptoContext& ctx, const Params& params,
                                                 ClientSecretMaterial& secret, LatencyRecorder& rec) {
    {
        ScopedTimer t(rec, "client_setup");
        secret = generate_client_secret_material(ctx, params);
    }

    SeededClientPublicMaterial eval_wire;
    {
        ScopedTimer t(rec, "client eval key generation (seeded)");
        eval_wire = build_seeded_public_material(ctx, secret);
    }

    ClientPublicMaterial pub;
    {
        ScopedTimer t(rec, "eval key unpacking");
        pub = reconstruct_public_material(ctx, params, eval_wire); // sequential -- see file header
    }
    return pub;
}

void run_one_query(const CryptoContext& ctx, const Params& params, ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec) {
    if (params.num_servers <= 0 || params.clusters_per_server <= 0) {
        throw std::invalid_argument("params.num_servers and params.clusters_per_server must be > 0 "
                                     "for the distributed benchmark; check derive_dependent_parameters().");
    }

    // --- client_query_gen: FULL query -- the client doesn't know server
    // topology, so it builds all num_clusters selector entries, not just
    // clusters_per_server. -----------------------------------------------------
    SeededQuery query_wire;
    {
        ScopedTimer t(rec, "client_query_gen (seeded)");

        std::vector<int64_t> embedding_values;
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            embedding_values.push_back(sample_signed_mod_value(params, rng));
        }

        query_wire = build_seeded_query(ctx, secret, embedding_values, params.num_clusters,
                                         params.desired_cluster_index);
    }

    std::vector<RLWECT> final_result;
    ReconstructedQuery query; // FULL: l embedding_cts, num_clusters selector_cts
    std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval;
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct;

    {
        ScopedTimer t(rec, "server_processing");

        // --- Unpacking: FULL, sequential. Cost does NOT scale down with
        // num_servers -- see file header for why. -------------------------------
        {
            ScopedTimer t_unpack(rec, "query unpacking");
            query = reconstruct_query(ctx, params, query_wire);
        }

        // --- RLWE switching: FULL embedding, multithreaded. Unaffected by
        // server topology, same as the non-distributed benchmarks. ---------------
        query_eval.resize(query.embedding_cts.size());
        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t j = 0; j < static_cast<int64_t>(query.embedding_cts.size()); ++j) {
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, query.embedding_cts[static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
            }
        }

        // --- "Distribution": slice ONE contiguous batch of
        // clusters_per_server selector ciphertexts out of the fully-
        // reconstructed set -- no extra work, just indexing. The batch
        // containing desired_cluster_index is picked for realism; which
        // batch doesn't affect timing. Falls back to batch 0 if
        // num_clusters doesn't divide evenly and desired_cluster_index
        // lands in the leftover region. -------------------------------------------
        int64_t batch_index = params.desired_cluster_index / params.clusters_per_server;
        if (batch_index >= params.num_servers) {
            batch_index = 0;
        }
        int64_t batch_start = batch_index * params.clusters_per_server;

        // --- RGSW switching: only this ONE batch, multithreaded -- stands
        // in for one server's latency; every server does the same amount of
        // work, in parallel, at the same time. -------------------------------------
        rgsw_ct.resize(static_cast<size_t>(params.clusters_per_server));
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching (one server's share)");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t c = 0; c < params.clusters_per_server; ++c) {
                const auto& gadget_ct = query.selector_cts[static_cast<size_t>(batch_start + c)];
                RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);
                rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
            }
        }

        // --- Scoring: only this ONE batch's clusters_per_server clusters. --------
        std::vector<RLWECT> partial_result; // one server's contribution, per split
        {
            ScopedTimer t_scoring(rec, "scoring calculations (one server's share)");

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

            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t c = 0; c < params.clusters_per_server; ++c) {
                    partial_result[static_cast<size_t>(s)].add(
                        partial_result[static_cast<size_t>(s)],
                        *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
                }
            }
        }

        // --- Cross-server summation: genuinely repeated num_servers times. -------
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

    std::string params_source = (argc >= 2) ? argv[1] : "Params::make_benchmark_params() (built-in defaults)";
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
    std::cout << "num_servers: " << params.num_servers << ", clusters_per_server: "
              << params.clusters_per_server << " (of " << params.num_clusters << " total clusters)\n\n";
    print_params(std::cout, params, params_source);

    std::mt19937_64 rng(std::random_device{}());

    LatencyRecorder rec;
    ClientSecretMaterial secret;

    std::cout << "Warming up client setup/registration (" << kSetupWarmupRuns << " runs)...\n";
    for (int i = 0; i < kSetupWarmupRuns; ++i) run_setup_and_registration(ctx, params, secret, rec);
    rec.clear();

    std::cout << "Measuring client setup/registration (" << kSetupMeasuredRuns << " runs)...\n";
    ClientPublicMaterial pub;
    for (int i = 0; i < kSetupMeasuredRuns; ++i) pub = run_setup_and_registration(ctx, params, secret, rec);

    std::cout << "\n=== Client setup / registration latency (seeded) ===\n";
    rec.print_summary();

    LatencyRecorder query_rec;

    std::cout << "\nWarming up per-query pipeline (" << kQueryWarmupRuns << " runs)...\n";
    for (int i = 0; i < kQueryWarmupRuns; ++i) run_one_query(ctx, params, secret, pub, rng, query_rec);
    query_rec.clear();

    std::cout << "Measuring per-query pipeline (" << kQueryMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kQueryMeasuredRuns; ++i) run_one_query(ctx, params, secret, pub, rng, query_rec);

    std::cout << "\n=== Per-query latency (simulated " << params.num_servers << "-server deployment, seeded, "
              << "compute-only, no network cost) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        out << "num_servers: " << params.num_servers << ", clusters_per_server: "
            << params.clusters_per_server << " (of " << params.num_clusters << " total clusters)\n";
        out << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
        print_params(out, params, params_source);
        out << "=== Client setup / registration latency (seeded) ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (seeded, compute-only, no network cost) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}