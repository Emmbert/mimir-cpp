// benchmark_latency_seeded_pool.cpp
//
// Seeded, single-threaded latency benchmark, POOL variant. Instead of the
// full database, a small fixed pool of kDatabasePoolSize distinct entries is
// built and reused across all (cluster,split) pairs, bounding peak memory for
// parameter sets whose full database won't fit in RAM.
//
// CAVEAT for reporting: the reused pool stays cache-hot, so "scoring
// calculations" here UNDER-reports the realistic (full-DB) scoring latency.
// Prefer benchmark_latency_seeded_fulldb.cpp; use this only when that OOMs.
//
// Run directly (NOT via ctest, which would swallow the printed table):
//   OMP_NUM_THREADS=1 ./benchmark_latency_seeded_pool
//   OMP_NUM_THREADS=1 ./benchmark_latency_seeded_pool params.json 1

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <filesystem>

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
constexpr int kSetupMeasuredRuns = 10;

constexpr int kQueryWarmupRuns = 2;
constexpr int kQueryMeasuredRuns = 50;

    std::filesystem::path compute_output_path(const std::string& params_arg) {
        const std::string base_name = "benchmark_latency_seeded_pool_results.txt";

        if (params_arg.empty()) {
            return std::filesystem::path(base_name);
        }

        std::filesystem::path param_path(params_arg);
        std::string stem = param_path.stem().string(); // "mimirI.json" -> "mimirI"

        std::filesystem::create_directories(stem); // no-op if it already exists
        return std::filesystem::path(stem) / base_name;
    }

// Small fixed pool, laid out [pool_idx][ring][j] so db_pool[idx][ring] is
// directly the length-l vector compute_split_score wants -- no per-j gather.
constexpr int64_t kDatabasePoolSize = 32;
using DbPool = std::vector<std::vector<std::vector<DatabasePolynomialEvalForm>>>; // [pool_idx][ring][j]

RLWECT compute_split_score(const CryptoContext& ctx, const std::vector<RLWECTEvalForm>& query_eval,
                            const std::vector<DatabasePolynomialEvalForm>& db_split) {
    RLWECTEvalForm score_eval(ctx.rlwe_param);
    for (size_t j = 0; j < query_eval.size(); ++j) {
        RLWECTEvalForm product_eval(ctx.rlwe_param);
        query_eval[j].mul(product_eval, *db_split[j].poly_eval);
        score_eval.add(score_eval, product_eval);
    }
    return RLWECT(score_eval);
}

/// Untimed preprocessing. Builds kDatabasePoolSize distinct entries, freeing
/// each polynomial's raw_values (test-only, never read during scoring).
DbPool build_database_pool(const CryptoContext& ctx, const Params& params, std::mt19937_64& rng) {
    int64_t r = params.num_component_rings;
    DbPool pool(static_cast<size_t>(kDatabasePoolSize));
    for (int64_t p = 0; p < kDatabasePoolSize; ++p) {
        auto& entry = pool[static_cast<size_t>(p)]; // [ring][j]
        entry.resize(static_cast<size_t>(r));
        for (int64_t ring = 0; ring < r; ++ring) {
            entry[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.embedding_length));
        }
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            std::vector<int64_t> raw(static_cast<size_t>(params.n));
            for (int64_t i = 0; i < params.n; ++i) {
                raw[static_cast<size_t>(i)] = sample_signed_value(params, rng).raw;
            }
            std::vector<DatabasePolynomialEvalForm> split =
                crt_split_database_polynomial_eval_form(ctx, params, raw); // one entry per ring
            for (int64_t ring = 0; ring < r; ++ring) {
                split[static_cast<size_t>(ring)].raw_values.clear();
                split[static_cast<size_t>(ring)].raw_values.shrink_to_fit();
                entry[static_cast<size_t>(ring)].push_back(std::move(split[static_cast<size_t>(ring)]));
            }
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

    DbPool db_pool = build_database_pool(ctx, params, rng); // untimed

    SeededQuery query_wire;
    std::vector<int64_t> embedding_values;
    embedding_values.reserve(static_cast<size_t>(params.embedding_length));
    for (int64_t j = 0; j < params.embedding_length; ++j) {
        embedding_values.push_back(sample_signed_mod_value(params, rng));
    }
    {
        ScopedTimer t(rec, "client_query_gen (seeded)");
        query_wire = build_seeded_query(ctx, params, secret, embedding_values, params.desired_cluster_index);
    }

    std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r));
    ReconstructedQuery query;
    std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r));
    std::vector<RLWEGadgetCT> rgsw_ct;
    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_unpack(rec, "query unpacking"); // uses OpenMP internally when r>1 -- see file header
            query = reconstruct_query(ctx, params, query_wire);
        }

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            for (int64_t ring = 0; ring < r; ++ring) {
                query_eval[static_cast<size_t>(ring)].reserve(query.embedding_cts[static_cast<size_t>(ring)].size());
                for (const auto& lwe_ct : query.embedding_cts[static_cast<size_t>(ring)]) {
                    RLWECT rlwe_ct(ctx.rlwe_param);
                    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                    query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct);
                }
            }
        }
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching");
            rgsw_ct.reserve(query.selector_cts.size());
            for (const auto& gadget_ct : query.selector_cts) {
                rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
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

            using Clock = std::chrono::steady_clock;
            std::chrono::duration<double, std::milli> score_time{0};
            std::chrono::duration<double, std::milli> mask_time{0};
            std::chrono::duration<double, std::milli> sum_time{0};

            for (int64_t c = 0; c < params.num_clusters; ++c) {
                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    int64_t pool_idx = (c * params.splits_per_cluster + s) % kDatabasePoolSize;
                    for (int64_t ring = 0; ring < r; ++ring) {
                        // Direct const ref into the pool -- raw_values were freed at build time.
                        const std::vector<DatabasePolynomialEvalForm>& db_for_ring =
                            db_pool[static_cast<size_t>(pool_idx)][static_cast<size_t>(ring)];

                        auto ts0 = Clock::now();
                        RLWECT score = compute_split_score(ctx, query_eval[static_cast<size_t>(ring)], db_for_ring);
                        auto ts1 = Clock::now();

                        RLWECT masked(ctx.rlwe_param);
                        rgsw_ct[static_cast<size_t>(c)].mul(masked, score);
                        auto ts2 = Clock::now();

                        final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)].add(
                            final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)], masked);
                        auto ts3 = Clock::now();

                        score_time += (ts1 - ts0);
                        mask_time += (ts2 - ts1);
                        sum_time += (ts3 - ts2);
                    }
                }
            }

            rec.add_sample("  -> score computation", score_time.count());
            rec.add_sample("  -> RGSW masking", mask_time.count());
            rec.add_sample("  -> cross-cluster summation", sum_time.count());
        }
    }

    {
        ScopedTimer t(rec, "client_decrypt");
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            std::vector<Vector> decrypted_per_ring;
            decrypted_per_ring.reserve(static_cast<size_t>(r));
            for (int64_t ring = 0; ring < r; ++ring) {
                decrypted_per_ring.push_back(secret.rlwe_sk->decrypt_vector(
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
                    ctx.component_encodings[static_cast<size_t>(ring)]));
            }
            if (r == 2) {
                for (int64_t i = 0; i < params.n; ++i) {
                    [[maybe_unused]] int64_t recomposed =
                        crt_recompose(decrypted_per_ring[0][i], decrypted_per_ring[1][i], params.comp_ring_modulus);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    Params params = load_benchmark_params_from_args(argc, argv, false);
    CryptoContext ctx = CryptoContext::from_params(params);

    std::string params_source = (argc >= 2) ? argv[1] : "Params::make_benchmark_params() (built-in defaults)";
    print_params(std::cout, params, params_source);

    const std::string params_arg = (argc >= 2) ? argv[1] : "";
    const std::filesystem::path kOutputFilePath = compute_output_path(params_arg);

    if (params.num_component_rings > 1) {
        std::cout << "NOTE: num_component_rings=" << params.num_component_rings
                  << " -- reconstruct_query uses OpenMP internally for CRT stream unpacking. "
                  << "Run with OMP_NUM_THREADS=1 for genuinely sequential timing.\n\n";
    }

    std::mt19937_64 rng(std::random_device{}());

    LatencyRecorder rec;
    ClientSecretMaterial secret;

    std::cout << "Warming up client setup/registration (" << kSetupWarmupRuns << " runs)...\n";
    for (int i = 0; i < kSetupWarmupRuns; ++i) {
        run_setup_and_registration(ctx, params, secret, rec);
    }
    rec.clear();

    std::cout << "Measuring client setup/registration (" << kSetupMeasuredRuns << " runs)...\n";
    ClientPublicMaterial pub;
    for (int i = 0; i < kSetupMeasuredRuns; ++i) {
        pub = run_setup_and_registration(ctx, params, secret, rec);
    }

    std::cout << "\n=== Client setup / registration latency (seeded) ===\n";
    rec.print_summary();

    LatencyRecorder query_rec;

    std::cout << "\nWarming up per-query pipeline (" << kQueryWarmupRuns << " runs)...\n";
    for (int i = 0; i < kQueryWarmupRuns; ++i) {
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }
    query_rec.clear();

    std::cout << "Measuring per-query pipeline (" << kQueryMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kQueryMeasuredRuns; ++i) {
        std::cout << "Iteration " << i << " " << std::flush;
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }

    std::cout << "\n=== Per-query latency (seeded, database pool) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        print_params(out, params, params_source);
        out << "=== Client setup / registration latency (seeded) ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (seeded, database pool) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}
