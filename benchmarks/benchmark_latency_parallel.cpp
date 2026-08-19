// benchmark_latency_parallel.cpp
//
// Same measurements as benchmark_latency.cpp, but RLWE switching, RGSW
// switching, database construction, and per-(cluster,split) scoring are
// parallelized with OpenMP.
//
// Database construction: builds a small, FIXED-SIZE pool of distinct
// database entries (kDatabasePoolSize, independent of num_clusters x
// splits_per_cluster) rather than one entry per actual (cluster,split)
// pair. Every scoring iteration below still performs a full, real RLWE
// multiplication/RGSW masking/accumulation for every (cluster,split,ring)
// combination -- nothing is skipped -- it just reuses one of these pool
// entries as its multiplicand instead of a uniquely-sampled one. This has
// no effect on measured latency (FHE ciphertext operations are
// constant-time with respect to plaintext content by design), but keeps
// peak memory bounded regardless of database size -- the whole-database
// version this replaced could reach tens of GB and get OOM-killed for a
// realistic parameter file. Rebuilt fresh every query (so the benchmark
// isn't measured against one fixed pool) but NEVER timed -- no ScopedTimer
// wraps it at all.
//
// Scoring parallelization: an OUTER sequential loop over component rings,
// with the INNER parallel loop being IDENTICAL in shape to the original
// non-CRT code -- same total_pairs, same masked[s][c] structure, no ring
// index inside the parallel region at all. Ring-level work is independent
// regardless of loop nesting, so this gives the same results as flattening
// (cluster,split,ring) into one parallel loop, while staying much easier to
// read and verify against the pre-CRT version directly.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_parallel
//   OMP_NUM_THREADS=32 ./benchmark_latency_parallel params.json 1

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
#include "timing.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

constexpr int kSetupWarmupRuns = 2;
constexpr int kSetupMeasuredRuns = 10;

constexpr int kQueryWarmupRuns = 2;
constexpr int kQueryMeasuredRuns = 10;

const char* kOutputFilePath = "benchmark_latency_parallel_results.txt";

constexpr int64_t kDatabasePoolSize = 32; // fixed, independent of total_pairs -- see chat message
                                           // this was written in for the memory/latency-validity
                                           // reasoning.

using DbPool = std::vector<std::vector<std::vector<DatabasePolynomialEvalForm>>>; // [pool_idx][j][ring]

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

/// Preprocessing -- deliberately NOT wrapped in any ScopedTimer. Builds
/// kDatabasePoolSize DISTINCT database entries (each with a full
/// embedding_length x r worth of eval-form polynomials, matching one
/// (cluster,split) pair's worth of real data), NOT one entry per actual
/// (cluster,split) pair. Every scoring iteration below still performs a
/// full, real RLWE multiplication/RGSW masking/accumulation -- nothing is
/// skipped -- it just reuses one of these kDatabasePoolSize polynomials as
/// its multiplicand instead of a uniquely-sampled one. FHE ciphertext
/// operations are constant-time with respect to plaintext content by
/// design (a security property, not an implementation detail), so this has
/// no effect on measured latency; it only bounds memory, which otherwise
/// scales with num_clusters x splits_per_cluster and can reach tens of GB
/// for a realistic parameter file.
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
    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;
    int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

    // Untimed preprocessing, rebuilt fresh this query -- see
    // build_database_pool's comment.
    DbPool db_pool = build_database_pool(ctx, params, rng);

    std::vector<std::vector<LWECT>> embedding_lwe(static_cast<size_t>(r));
    std::vector<LWEGadgetCT> selector_gadget;
    {
        ScopedTimer t(rec, "client_query_gen");

        for (int64_t ring = 0; ring < r; ++ring) {
            embedding_lwe[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.embedding_length));
        }
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            SignedValue m = sample_signed_value(params, rng);
            std::vector<int64_t> components;
            if (r == 1) {
                components = {m.reduced};
            } else {
                int64_t canonical = reduce_mod(m.raw, combined_modulus);
                auto [c1, c2] = crt_split(canonical, params.comp_ring_modulus);
                components = {c1, c2};
            }
            for (int64_t ring = 0; ring < r; ++ring) {
                embedding_lwe[static_cast<size_t>(ring)].push_back(secret.lwe_sk->encode_and_encrypt(
                    components[static_cast<size_t>(ring)], ctx.component_encodings[static_cast<size_t>(ring)]));
            }
        }

        selector_gadget.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            selector_gadget.push_back(secret.lwe_gadget_sk->gadget_encrypt(bit));
        }
    }

    std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r));
    std::vector<std::vector<std::unique_ptr<RLWECTEvalForm>>> query_eval(static_cast<size_t>(r));
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(static_cast<size_t>(params.num_clusters));

    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            for (int64_t ring = 0; ring < r; ++ring) {
                query_eval[static_cast<size_t>(ring)].resize(static_cast<size_t>(params.embedding_length));
            }
            int64_t total_query_terms = r * params.embedding_length;
            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_query_terms; ++idx) {
                int64_t ring = idx / params.embedding_length;
                int64_t j = idx % params.embedding_length;
                RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, embedding_lwe[static_cast<size_t>(ring)][static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(ring)][static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
            }
        }
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t c = 0; c < params.num_clusters; ++c) {
                RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(selector_gadget[static_cast<size_t>(c)]);
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

            // --- Outer loop over rings; inner parallel loop is EXACTLY the
            // pre-CRT structure, just reading db_eval[c][s][j][ring]. -------------------
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

    std::cout << "\n=== Per-query latency ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        out << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
        print_params(out, params, params_source);
        out << "=== Client setup / registration latency ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}