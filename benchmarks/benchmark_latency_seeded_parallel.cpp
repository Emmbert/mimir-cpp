// benchmark_latency_seeded_parallel.cpp
//
// Combines benchmark_latency_seeded.cpp (seeded eval keys AND seeded query,
// both reconstructed with NO secret key involved in reconstruction) with
// benchmark_latency_parallel.cpp's threading pattern.
//
// Unpacking (reconstruct_public_material, reconstruct_query) stays
// SEQUENTIAL -- rejection sampling inside SeededUniformDistribution makes
// the byte offset each ciphertext starts at data-dependent, so there's no
// way to parallelize it without changing the sampling method itself (see
// the earlier discussion). Only the ACTUAL switching (RLWE, RGSW) and
// per-(cluster,split) scoring are parallelized, exactly as in
// benchmark_latency_parallel.cpp.
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

const char* kOutputFilePath = "benchmark_latency_seeded_parallel_results.txt";

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
    ReconstructedQuery query;
    std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval;
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct;

    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_unpack(rec, "query unpacking"); // sequential -- see file header
            query = reconstruct_query(ctx, params, query_wire);
        }

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

            final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result.emplace_back(ctx.rlwe_param);
            }

            std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
                static_cast<size_t>(params.splits_per_cluster));
            for (auto& row : masked) {
                row.resize(static_cast<size_t>(params.num_clusters));
            }

            int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

            std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
            for (auto& r : thread_rngs) r.seed(rng());

            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_pairs; ++idx) {
                int64_t c = idx / params.splits_per_cluster;
                int64_t s = idx % params.splits_per_cluster;
                std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

                std::vector<DatabasePolynomialEvalForm> db_split;
                db_split.reserve(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    db_split.push_back(build_random_database_polynomial_eval_form(ctx, params, local_rng));
                }

                RLWECT score = compute_split_score(ctx, query_eval, db_split);

                RLWECT masked_val(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);

                masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);
            }

            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t c = 0; c < params.num_clusters; ++c) {
                    final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)],
                                                               *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
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
