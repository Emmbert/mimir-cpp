// benchmark_latency_seeded_parallel.cpp
//
// Combines benchmark_latency_seeded.cpp with benchmark_latency_parallel.cpp's
// threading pattern: outer sequential loop over rings, inner parallel loop
// identical to the pre-CRT structure. Database rebuilt fresh every query,
// never timed. query.embedding_cts is already [ring][j] (reconstruct_query
// is CRT-aware), so no manual per-ring encryption loop is needed.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_seeded_parallel
//   OMP_NUM_THREADS=16 ./benchmark_latency_seeded_parallel params.json 1

#include <omp.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "crt.hpp"
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
constexpr int kSetupMeasuredRuns = 20;

constexpr int kQueryWarmupRuns = 2;
constexpr int kQueryMeasuredRuns = 20;

const char* kOutputFilePath = "benchmark_latency_seeded_parallel_results.txt";

constexpr int64_t kDatabasePoolSize = 32; // fixed, independent of total_pairs -- see
                                           // benchmark_latency_parallel.cpp's header for the
                                           // memory/latency-validity reasoning.

using DbPool = std::vector<std::vector<std::vector<DatabasePolynomialEvalForm>>>; // [pool_idx][j][ring]

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

/// Untimed preprocessing -- see benchmark_latency_parallel.cpp's
/// build_database_pool for the full reasoning.
DbPool build_database_pool(const CryptoContext& ctx, const Params& params, std::mt19937_64& rng) {
    DbPool pool(static_cast<size_t>(kDatabasePoolSize));

    std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
    for (auto& tr : thread_rngs) tr.seed(rng());

    #pragma omp parallel for schedule(dynamic)
    for (int64_t p = 0; p < kDatabasePoolSize; ++p) {
        std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

        auto& entry = pool[static_cast<size_t>(p)];
        entry.resize(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            std::vector<int64_t> raw(static_cast<size_t>(params.n));
            for (int64_t i = 0; i < params.n; ++i) {
                raw[static_cast<size_t>(i)] = sample_signed_value(params, local_rng).raw;
            }
            entry[static_cast<size_t>(j)] = crt_split_database_polynomial_eval_form(ctx, params, raw);
        }
    }
    return pool;
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
        pub = reconstruct_public_material(ctx, params, eval_wire);
    }
    return pub;
}

void run_one_query(const CryptoContext& ctx, const Params& params, ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec) {
    int64_t r = params.num_component_rings;
    int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

    DbPool db_pool = build_database_pool(ctx, params, rng); // untimed

    SeededQuery query_wire;
    {
        std::vector<int64_t> embedding_values;
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            embedding_values.push_back(sample_signed_mod_value(params, rng));
        }
        ScopedTimer t(rec, "client_query_gen (seeded)");
        query_wire = build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);
    }

    std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r));
    ReconstructedQuery query;
    std::vector<std::vector<std::unique_ptr<RLWECTEvalForm>>> query_eval(static_cast<size_t>(r));
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct;

    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_unpack(rec, "query unpacking");
            query = reconstruct_query(ctx, params, query_wire);
        }

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            for (int64_t ring = 0; ring < r; ++ring) {
                query_eval[static_cast<size_t>(ring)].resize(query.embedding_cts[static_cast<size_t>(ring)].size());
            }
            int64_t total_query_terms = r * params.embedding_length;
            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_query_terms; ++idx) {
                int64_t ring = idx / params.embedding_length;
                int64_t j = idx % params.embedding_length;
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(
                    rlwe_ct, query.embedding_cts[static_cast<size_t>(ring)][static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(ring)][static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
            }
        }

        rgsw_ct.resize(query.selector_cts.size());
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t c = 0; c < static_cast<int64_t>(query.selector_cts.size()); ++c) {
                RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(query.selector_cts[static_cast<size_t>(c)]);
                rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
            }
        }

        {
            ScopedTimer t_scoring(rec, "scoring calculations");

            for (int64_t ring = 0; ring < r; ++ring) {
                final_result[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.splits_per_cluster));
                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    final_result[static_cast<size_t>(ring)].emplace_back(ctx.rlwe_param);
                }
            }

            // Outer loop over rings; inner parallel loop identical to the pre-CRT structure.
            for (int64_t ring = 0; ring < r; ++ring) {
                std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
                    static_cast<size_t>(params.splits_per_cluster));
                for (auto& row : masked) {
                    row.resize(static_cast<size_t>(params.num_clusters));
                }

                #pragma omp parallel for schedule(dynamic)
                for (int64_t idx = 0; idx < total_pairs; ++idx) {
                    int64_t c = idx / params.splits_per_cluster;
                    int64_t s = idx % params.splits_per_cluster;

                    const auto& slot = db_pool[static_cast<size_t>(idx % kDatabasePoolSize)];
                    std::vector<DatabasePolynomialEvalForm> db_split;
                    db_split.reserve(static_cast<size_t>(params.embedding_length));
                    for (int64_t j = 0; j < params.embedding_length; ++j) {
                        db_split.push_back(slot[static_cast<size_t>(j)][static_cast<size_t>(ring)]);
                    }

                    RLWECT score = compute_split_score(ctx, query_eval[static_cast<size_t>(ring)], db_split);

                    RLWECT masked_val(ctx.rlwe_param);
                    rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);

                    masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);
                }

                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    for (int64_t c = 0; c < params.num_clusters; ++c) {
                        final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)].add(
                            final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                            *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
                    }
                }
            }
        }
    }

    {
        ScopedTimer t(rec, "client_decrypt");

        // Decrypt every (split, ring) pair in parallel -- each thread
        // writes its own (s, ring) slot, no shared state touched more than
        // once. Structurally the same safety argument already relied on
        // for RLWE/RGSW switching elsewhere in this file (read secret/
        // public key material from multiple threads, no mutation) -- worth
        // noting this specific call hasn't been exercised concurrently
        // before, unlike the switching functions.
        std::vector<std::vector<Vector>> decrypted(static_cast<size_t>(params.splits_per_cluster)); // [s][ring]
        for (auto& row : decrypted) {
            row.resize(static_cast<size_t>(r));
        }

        int64_t total_decrypts = params.splits_per_cluster * r;
        #pragma omp parallel for schedule(dynamic)
        for (int64_t idx = 0; idx < total_decrypts; ++idx) {
            int64_t s = idx / r;
            int64_t ring = idx % r;
            decrypted[static_cast<size_t>(s)][static_cast<size_t>(ring)] = secret.rlwe_sk->decrypt_vector(
                final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                ctx.component_encodings[static_cast<size_t>(ring)]);
        }

        // Recompose every coefficient of every split in parallel -- pure
        // arithmetic, zero library calls, zero shared state (results are
        // discarded, matching "benchmarks measure timing only").
        if (r == 2) {
            #pragma omp parallel for schedule(dynamic) collapse(2)
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t i = 0; i < params.n; ++i) {
                    [[maybe_unused]] int64_t recomposed = crt_recompose(
                        decrypted[static_cast<size_t>(s)][0][i], decrypted[static_cast<size_t>(s)][1][i],
                        params.comp_ring_modulus);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string params_source;
    Params params = load_benchmark_params_from_args(argc, argv, /*distributed=*/false, &params_source);
    CryptoContext ctx = CryptoContext::from_params(params);
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
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

    std::cout << "\n=== Per-query latency (seeded) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        out << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
        print_params(out, params, params_source);
        out << "=== Client setup / registration latency (seeded) ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (seeded) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}