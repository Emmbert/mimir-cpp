// benchmark_latency_parallel.cpp
//
// Same measurements as benchmark_latency.cpp, but RLWE switching, RGSW
// switching, and the per-(cluster,split) scoring work (db build + score +
// RGSW mask) are parallelized with OpenMP.
//
// Why threads, not processes: this is a single-machine benchmark. Processes
// would need real IPC/serialization to move ciphertexts between them, which
// either doesn't reflect a real colocated deployment (extra cost that
// wouldn't exist there) or doesn't reflect a real distributed one either
// (localhost IPC != real network latency) -- see the much earlier discussion
// on this. Threads give a clean measurement of "how much do more cores
// help", nothing more, nothing less.
//
// Why the cross-cluster sum is NOT parallelized: every cluster for a given
// split writes into the SAME final_result[s] accumulator. Concurrent
// final_result[s].add(final_result[s], masked) calls from multiple threads
// would race. Instead, each thread writes its (cluster, split) result into
// its OWN slot of a pre-sized 2D array (no aliasing -- safe), and the actual
// summation over clusters happens sequentially afterward. That sum is cheap
// (RLWECT::add is a plain elementwise mod-add) relative to the multiply/mask
// work, so parallelizing it isn't worth the complexity.
//
// A note for this specific CPU (AMD Ryzen 9 7950X3D): it has two asymmetric
// CCDs (one with 3D V-Cache and a lower clock, one without and a higher
// clock). schedule(dynamic) is used below instead of the OpenMP default
// schedule(static) specifically because equal-sized static chunks would
// finish at different times across the two CCDs, leaving faster cores idle.
// Also worth comparing OMP_NUM_THREADS=16 (one thread per physical core)
// against 32 (full SMT) empirically -- this workload is ALU/cache-bound NTT
// and gadget arithmetic, which doesn't reliably benefit from SMT the way
// memory-stall-heavy code does.
//
// Run directly, with the desired thread count set via the environment:
//   OMP_NUM_THREADS=16 ./benchmark_latency_parallel
//   OMP_NUM_THREADS=32 ./benchmark_latency_parallel

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

const char* kOutputFilePath = "benchmark_latency_parallel_results.txt";

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
    // --- client_query_gen: sequential -- 25ms out of ~5.5s total isn't
    // worth parallelizing, and it's a client-side, single-request operation
    // in practice, not a batch job. ------------------------------------------
    std::vector<LWECT> embedding_lwe;
    std::vector<LWEGadgetCT> selector_gadget;
    {
        ScopedTimer t(rec, "client_query_gen");

        embedding_lwe.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            int64_t m = sample_signed_mod_value(params, rng);
            embedding_lwe.push_back(secret.lwe_sk->encode_and_encrypt(m, ctx.encoding));
        }

        selector_gadget.reserve(static_cast<size_t>(params.num_clusters));
        for (int64_t c = 0; c < params.num_clusters; ++c) {
            int64_t bit = (c == params.desired_cluster_index) ? 1 : 0;
            selector_gadget.push_back(secret.lwe_gadget_sk->gadget_encrypt(bit));
        }
    }

    std::vector<RLWECT> final_result;
    // unique_ptr slots: each index written by exactly one thread, so no
    // aliasing even though the vector itself is shared -- safe as long as
    // it's pre-sized (resize) BEFORE the parallel region, so no reallocation
    // happens concurrently with writes.
    std::vector<std::unique_ptr<RLWECTEvalForm>> query_eval(static_cast<size_t>(params.embedding_length));
    std::vector<std::unique_ptr<RLWEGadgetCT>> rgsw_ct(static_cast<size_t>(params.num_clusters));

    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            #pragma omp parallel for schedule(dynamic)
            for (int64_t j = 0; j < params.embedding_length; ++j) {
                RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, embedding_lwe[static_cast<size_t>(j)]);
                query_eval[static_cast<size_t>(j)] = std::make_unique<RLWECTEvalForm>(rlwe_ct);
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

            final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result.emplace_back(ctx.rlwe_param);
            }

            // Each thread computes one (cluster, split) pair's masked score
            // and writes it into its own [s][c] slot -- no shared mutable
            // state touched inside the parallel region.
            //
            // Built via resize(), not the vector(count, prototype) fill
            // constructor: that constructor needs to COPY the prototype
            // `count` times, and unique_ptr's copy constructor is deleted --
            // exactly the error this triggers. Default-constructing the
            // outer vector (empty inner vectors, no copy involved) then
            // resizing each row separately (default-constructs new
            // unique_ptr elements, also no copy involved) avoids it.
            std::vector<std::vector<std::unique_ptr<RLWECT>>> masked(
            static_cast<size_t>(params.splits_per_cluster));
            for (auto& row : masked) {
                row.resize(static_cast<size_t>(params.num_clusters));
            }

            // Flatten (cluster, split) into one index for a single parallel
            // loop -- gives OpenMP more, smaller chunks to balance across
            // the two asymmetric CCDs than a collapse(2) over uneven-sized
            // dimensions would.
            int64_t total_pairs = params.num_clusters * params.splits_per_cluster;

            // Each thread needs its own rng -- std::mt19937_64 is not
            // thread-safe to share. Seed each thread's rng independently
            // from the (single-threaded) outer rng, once, before the
            // parallel region.
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

            // Sequential reduction: cheap relative to the parallel work
            // above, and avoids any need for a custom OpenMP reduction on a
            // non-primitive type.
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

    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n\n";

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
