// benchmark_latency.cpp
//
// Single-machine, single-threaded latency benchmark. Two separately-measured
// things, matching how a real deployment actually amortizes cost:
//
//   1. Client setup + registration -- happens ONCE per client session, so
//      measured in its own loop (fresh client each repetition), reported
//      separately from per-query cost.
//   2. Per-query cost -- ONE client is set up (untimed here, since it's
//      already measured above), then many queries are issued against it,
//      each broken into:
//        - client_query_gen: encrypting the embedding vector + cluster
//          selector bits.
//        - server_processing: the full Step 5-7 pipeline, itself broken
//          into "RLWE ciphertext switching", "RGSW ciphertext switching",
//          and "scoring calculations" -- the last of which is further
//          broken into db polynomial building / score computation / RGSW
//          masking / cross-cluster summation (see note below on why those
//          four are accumulated differently from everything else).
//        - client_decrypt: decrypting the resulting splits_per_cluster
//          ciphertexts.
//
// A note on measurement overhead: every ScopedTimer construction/destruction
// costs roughly 100-300ns (two clock reads + a few hash-map lookups + a
// vector push in LatencyRecorder). That's negligible next to a substep that
// itself runs a loop of hundreds of iterations (RLWE switching, RGSW
// switching, scoring calculations as a whole) -- those substeps individually
// cost many microseconds to milliseconds, so a few hundred ns of bookkeeping
// is well under 1% overhead. It stops being negligible once you try to time
// something that ITSELF only costs a few microseconds -- e.g. a single
// cluster's db-polynomial build or a single RGSW mask, called ~845 times per
// query. Wrapping each of those individually in its own ScopedTimer would
// measurably inflate the numbers. Instead, for the four scoring sub-steps
// below, raw std::chrono durations are accumulated in plain local variables
// inside the loop (cheap: just two clock reads per iteration, no hash-map
// bookkeeping), and only ONE LatencyRecorder::add_sample call happens per
// query, after the loop, with the totals.
//
// Both top-level loops (setup/registration, per-query) have a discarded
// warm-up phase before the measured runs. Results are printed to stdout AND
// written to a file (see kOutputFilePath).
//
// Run directly (NOT via ctest, which would swallow the printed table):
//   ./benchmark_latency

#include <chrono>
#include <fstream>
#include <iostream>
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
constexpr int kQueryMeasuredRuns = 10; // each run does num_clusters RGSW switches --
                                       // this is the expensive part; raise once you
                                       // know how long one run takes on your machine.

const char* kOutputFilePath = "benchmark_latency_results.txt";

RLWECT switch_to_rlwe(const CryptoContext& ctx, const ClientPublicMaterial& pub, const LWECT& lwe_ct) {
    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    return rlwe_ct;
}

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

/// Runs one full client setup + registration, recording into `rec` under
/// "client_setup" and "client_registration".
void run_setup_and_registration(const CryptoContext& ctx, const Params& params, LatencyRecorder& rec) {
    ClientSecretMaterial secret;
    {
        ScopedTimer t(rec, "client_setup");
        secret = generate_client_secret_material(ctx, params);
        [[maybe_unused]] ClientPublicMaterial pub = generate_client_public_material(ctx, secret);
    }
}

/// Runs one full query against an already-set-up client, recording into
/// `rec` under "client_query_gen", "server_processing" (with sub-stages),
/// and "client_decrypt".
void run_one_query(const CryptoContext& ctx, const Params& params, const ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec) {
    // --- client_query_gen: encrypt embedding vector + cluster selector. ---
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

    // --- server_processing: the full Step 5-7 pipeline. -------------------
    std::vector<RLWECT> final_result;
    std::vector<RLWECTEvalForm> query_eval;
    std::vector<RLWEGadgetCT> rgsw_ct;
    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            query_eval.reserve(embedding_lwe.size());
            for (const auto& lwe_ct : embedding_lwe) {
                RLWECT rlwe_ct = switch_to_rlwe(ctx, pub, lwe_ct);
                query_eval.emplace_back(rlwe_ct);
            }
        }
        {
            ScopedTimer t_switch_rgsw(rec, "RGSW ciphertext switching");
            rgsw_ct.reserve(selector_gadget.size());
            for (const auto& gadget_ct : selector_gadget) {
                rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
            }
        }

        {
            ScopedTimer t_scoring(rec, "scoring calculations");

            final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result.emplace_back(ctx.rlwe_param);
            }

            // See the file header for why these four are accumulated in
            // plain local variables (cheap: two clock reads per iteration)
            // rather than each getting its own ScopedTimer per iteration
            // (which would cost more, per call, than the ~microsecond
            // operations being measured).
            using Clock = std::chrono::steady_clock;
            std::chrono::duration<double, std::milli> db_build_time{0};
            std::chrono::duration<double, std::milli> score_time{0};
            std::chrono::duration<double, std::milli> mask_time{0};
            std::chrono::duration<double, std::milli> sum_time{0};

            for (int64_t c = 0; c < params.num_clusters; ++c) {
                for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                    auto t0 = Clock::now();
                    std::vector<DatabasePolynomialEvalForm> db_split;
                    db_split.reserve(static_cast<size_t>(params.embedding_length));
                    for (int64_t j = 0; j < params.embedding_length; ++j) {
                        db_split.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
                    }
                    auto t1 = Clock::now();

                    RLWECT score = compute_split_score(ctx, query_eval, db_split);
                    auto t2 = Clock::now();

                    RLWECT masked(ctx.rlwe_param);
                    rgsw_ct[static_cast<size_t>(c)].mul(masked, score);
                    auto t3 = Clock::now();

                    final_result[static_cast<size_t>(s)].add(final_result[static_cast<size_t>(s)], masked);
                    auto t4 = Clock::now();

                    db_build_time += (t1 - t0);
                    score_time += (t2 - t1);
                    mask_time += (t3 - t2);
                    sum_time += (t4 - t3);
                }
            }

            // One sample per query for each sub-stage -- totals across every
            // (cluster, split) pair in this query, not a per-cluster average.
            rec.add_sample("  -> db polynomial building", db_build_time.count());
            rec.add_sample("  -> score computation", score_time.count());
            rec.add_sample("  -> RGSW masking", mask_time.count());
            rec.add_sample("  -> cross-cluster summation", sum_time.count());
        }
    }

    // --- client_decrypt: decrypt every split's result. ---------------------
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

    std::mt19937_64 rng(std::random_device{}());

    LatencyRecorder rec;

    // --- Setup + registration: fresh client each repetition. ---------------
    std::cout << "Warming up client setup/registration (" << kSetupWarmupRuns << " runs)...\n";
    for (int i = 0; i < kSetupWarmupRuns; ++i) {
        run_setup_and_registration(ctx, params, rec);
    }
    rec.clear();

    std::cout << "Measuring client setup/registration (" << kSetupMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kSetupMeasuredRuns; ++i) {
        run_setup_and_registration(ctx, params, rec);
    }

    std::cout << "\n=== Client setup / registration latency ===\n";
    rec.print_summary();

    // --- Per-query: ONE client set up (untimed), many queries against it. --
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

    LatencyRecorder query_rec;

    std::cout << "\nWarming up per-query pipeline (" << kQueryWarmupRuns << " runs)...\n";
    for (int i = 0; i < kQueryWarmupRuns; ++i) {
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }
    query_rec.clear();

    std::cout << "Measuring per-query pipeline (" << kQueryMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kQueryMeasuredRuns; ++i) {
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }

    std::cout << "\n=== Per-query latency ===\n";
    query_rec.print_summary();

    // --- Write both tables to a file too. -----------------------------------
    std::ofstream out(kOutputFilePath);
    if (out) {
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