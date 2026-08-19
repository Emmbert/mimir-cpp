// benchmark_latency_distributed.cpp
//
// Simulates a deployment with num_servers identical machines, COMPUTE-ONLY
// (no real processes, no network/IPC).
//
// CRT distribution model: the total work is num_clusters * splits_per_cluster
// * r "databases" (one per (cluster, split, ring) triple), spread across
// num_servers machines split into r equal groups -- one group per component
// ring (e.g. for r==2: half the machines process ONLY ZZ_p0 databases, the
// other half ONLY ZZ_p1). So machines_per_ring = num_servers / r machines
// are available for ANY ONE ring's share of the work, and one such machine
// handles clusters_per_machine = num_clusters / machines_per_ring whole
// clusters (all their splits) -- bigger than the old (non-CRT)
// clusters_per_server by a factor of r, since fewer machines are available
// per ring now.
//
// Only ONE representative computation is actually run (for one ring, one
// machine's share) and timed. That single result is then summed
// machines_per_ring times into final_result[0][s], and summed
// machines_per_ring times AGAIN into final_result[1][s] -- i.e. the SAME
// computed values stand in for both rings' cross-machine aggregation. This
// deliberately gives an incorrect VALUE (both rings end up holding the same
// underlying ZZ_p0 result) but a representative LATENCY, since every
// machine (regardless of which ring it's assigned to) does the same amount
// of work. Query building and the shared RLWE switch are NOT shortcut --
// those are real, full-cost client operations for both rings.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_distributed
//   OMP_NUM_THREADS=16 ./benchmark_latency_distributed params.json 8
// TODO embedding sampling should be moved out of the query generation time (because the rejection sampling takes longer
// TODO than the real embedding generation

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

const char* kOutputFilePath = "benchmark_latency_distributed_results.txt";

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
    if (params.num_servers <= 0) {
        throw std::invalid_argument("params.num_servers must be > 0.");
    }
    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    int64_t machines_per_ring = (params.num_servers + r - 1 ) / r; //ceil(params.num_servers / r);
    //if (machines_per_ring <= 0) machines_per_ring = 1;
    int64_t clusters_per_machine = (params.num_clusters + machines_per_ring -1 ) / machines_per_ring; // ceil(params.num_clusters / machines_per_ring);
    //if (clusters_per_machine <= 0) clusters_per_machine = 1;

    // --- client_query_gen: FULL query, all rings, all num_clusters selector
    // bits -- the client doesn't know server topology, and this is a real
    // client-side cost, not shortcut. ------------------------------------------------
    std::vector<std::vector<LWECT>> embedding_lwe(static_cast<size_t>(r)); // [ring][j]
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

    std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r)); // [ring][s]
    std::vector<std::vector<std::unique_ptr<RLWECTEvalForm>>> query_eval(static_cast<size_t>(r));
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct;

    {
        ScopedTimer t(rec, "server_processing");

        // --- Shared, non-sharded: embedding -> RLWE/eval, all rings, full,
        // real cost. ------------------------------------------------------------------
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
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(
                    rlwe_ct, embedding_lwe[static_cast<size_t>(ring)][static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(ring)][static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
            }
        }

        // --- ONE representative machine's share: clusters_per_machine whole
        // clusters, all splits, for ONE representative ring (ring 0). ----------------
        /*int64_t batch_index = params.desired_cluster_index / clusters_per_machine;
        if (batch_index >= machines_per_ring) {
            batch_index = 0;
        }
        int64_t batch_start = batch_index * clusters_per_machine;*/

        rgsw_ct.resize(static_cast<size_t>(clusters_per_machine));
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching (one machine's share)");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t c = 0; c < clusters_per_machine; ++c) {
                const auto& gadget_ct = selector_gadget[static_cast<size_t>(c)];
                RLWEGadgetCT rgsw = pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct);
                rgsw_ct[static_cast<size_t>(c)] = std::make_unique<RLWEGadgetCT>(std::move(rgsw));
            }
        }

        // Per-machine database shard (ring 0's modulus). UNTIMED preprocessing:
        // a real server builds its shard once at setup, never per query -- only
        // scoring against it is the online cost, so it must not sit in the timer.
        int64_t total_pairs = clusters_per_machine * params.splits_per_cluster;
        std::vector<std::vector<std::vector<DatabasePolynomialEvalForm>>> db_ring0( // [c_local][s][j]
            static_cast<size_t>(clusters_per_machine));
        for (auto& per_cluster : db_ring0) {
            per_cluster.resize(static_cast<size_t>(params.splits_per_cluster));
        }
        {
            std::vector<std::mt19937_64> thread_rngs(static_cast<size_t>(omp_get_max_threads()));
            for (auto& tr : thread_rngs) tr.seed(rng());

            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_pairs; ++idx) {
                int64_t c = idx / params.splits_per_cluster;
                int64_t s = idx % params.splits_per_cluster;
                std::mt19937_64& local_rng = thread_rngs[static_cast<size_t>(omp_get_thread_num())];

                auto& slot = db_ring0[static_cast<size_t>(c)][static_cast<size_t>(s)];
                slot.resize(static_cast<size_t>(params.embedding_length));
                for (int64_t j = 0; j < params.embedding_length; ++j) {
                    std::vector<int64_t> raw(static_cast<size_t>(params.n));
                    for (int64_t i = 0; i < params.n; ++i) {
                        raw[static_cast<size_t>(i)] = sample_signed_value(params, local_rng).raw;
                    }
                    slot[static_cast<size_t>(j)] = build_database_polynomial_eval_form_from_raw_values(
                        ctx, params, raw, component_ring_modulus(params, 0));
                }
            }
        }

        std::vector<RLWECT> partial_result;
        {
            ScopedTimer t_scoring(rec, "scoring calculations (one machine's share)");

            partial_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                partial_result.emplace_back(ctx.rlwe_param);
            }

            std::vector<std::vector<std::unique_ptr<RLWECT>>> masked( // [s][c_local]
                static_cast<size_t>(params.splits_per_cluster));
            for (auto& row : masked) {
                row.resize(static_cast<size_t>(clusters_per_machine));
            }

            #pragma omp parallel for schedule(dynamic)
            for (int64_t idx = 0; idx < total_pairs; ++idx) {
                int64_t c = idx / params.splits_per_cluster;
                int64_t s = idx % params.splits_per_cluster;

                RLWECT score = compute_split_score(ctx, query_eval[0], db_ring0[static_cast<size_t>(c)][static_cast<size_t>(s)]);

                RLWECT masked_val(ctx.rlwe_param);
                rgsw_ct[static_cast<size_t>(c)]->mul(masked_val, score);

                masked[static_cast<size_t>(s)][static_cast<size_t>(c)] = std::make_unique<RLWECT>(masked_val);
            }

            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                for (int64_t c = 0; c < clusters_per_machine; ++c) {
                    partial_result[static_cast<size_t>(s)].add(partial_result[static_cast<size_t>(s)],
                                                                *masked[static_cast<size_t>(s)][static_cast<size_t>(c)]);
                }
            }
        }

        // --- Cross-machine summation: the SAME partial_result stands in for
        // EVERY machine in EVERY ring's group -- machines_per_ring copies
        // summed into EACH ring's final accumulator. Parallelizable since
        // the r ring-accumulations are entirely independent of each other. -------------
        {
            ScopedTimer t_cross_server(rec, "cross-machine summation");

            for (int64_t ring = 0; ring < r; ++ring) {
                final_result[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.splits_per_cluster));
                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    final_result[static_cast<size_t>(ring)].emplace_back(ctx.rlwe_param);
                }
            }

            #pragma omp parallel for schedule(dynamic) collapse(2)
            for (int64_t ring = 0; ring < r; ++ring) {
                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    RLWECT acc(ctx.rlwe_param);
                    for (int64_t m = 0; m < machines_per_ring; ++m) {
                        acc.add(acc, partial_result[static_cast<size_t>(s)]);
                    }
                    final_result[static_cast<size_t>(ring)][static_cast<size_t>(s)] = acc;
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
    Params params = load_benchmark_params_from_args(argc, argv, true);
    CryptoContext ctx = CryptoContext::from_params(params);

    std::string params_source = (argc >= 2) ? argv[1] : "Params::make_benchmark_params() (built-in defaults)";
    int64_t r = params.num_component_rings;
    int64_t machines_per_ring = (params.num_servers + r - 1 ) / r;
    int64_t clusters_per_machine = (params.num_clusters + machines_per_ring - 1) / machines_per_ring; // ceil(params.num_clusters / (machines_per_ring > 0 ? machines_per_ring : 1));
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
    std::cout << "num_servers: " << params.num_servers << ", num_component_rings: " << r
              << ", machines_per_ring: " << machines_per_ring << ", clusters_per_machine: " << clusters_per_machine
              << " (of " << params.num_clusters << " total clusters)\n\n";
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
    for (int i = 0; i < kQueryMeasuredRuns; ++i){
        std::cout << "Iteration " << i << " " << std::flush;
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }

    std::cout << "\n=== Per-query latency (simulated " << params.num_servers << "-machine deployment, "
              << "compute-only, no network cost) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        out << "num_servers: " << params.num_servers << ", num_component_rings: " << r
            << ", machines_per_ring: " << machines_per_ring << ", clusters_per_machine: " << clusters_per_machine
            << " (of " << params.num_clusters << " total clusters)\n";
        out << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";
        print_params(out, params, params_source);
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