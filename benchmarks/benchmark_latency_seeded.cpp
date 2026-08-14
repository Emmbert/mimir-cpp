// benchmark_latency_seeded.cpp
//
// Same as benchmark_latency.cpp, but eval keys AND the query are both built
// with seeds, wire-compressed, and reconstructed with NO secret key
// involved in reconstruction -- see seeded_eval_keys.hpp and
// seeded_query.hpp. Unpacking (reconstruct_public_material,
// reconstruct_query) is timed as its own stage, separate from the actual
// switching -- this is the number the earlier "should we benchmark seed
// unpacking" discussion was about.
//
// Eval-key unpacking happens ONCE per client session (inside setup), not
// once per query -- matching a real deployment, where the server unpacks a
// client's eval keys once at registration and reuses the result for every
// subsequent query.
//
// Run directly (NOT via ctest, which would swallow the printed table):
//   ./benchmark_latency_seeded
//   ./benchmark_latency_seeded params.json 1

#include <chrono>
#include <fstream>
#include <iostream>
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

constexpr int kQueryWarmupRuns = 5;
constexpr int kQueryMeasuredRuns = 30;

const char* kOutputFilePath = "benchmark_latency_seeded_results.txt";

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

/// Client setup + seeded eval-key generation + unpacking. Returns the usable
/// (reconstructed) ClientPublicMaterial for the caller to hold onto and
/// reuse across many queries.
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
    SeededQuery query_wire;
    {
        ScopedTimer t(rec, "client_query_gen (seeded)");

        std::vector<int64_t> embedding_values;
        embedding_values.reserve(static_cast<size_t>(params.embedding_length));
        for (int64_t j = 0; j < params.embedding_length; ++j) {
            embedding_values.push_back(sample_signed_mod_value(params, rng));
        }
        query_wire = build_seeded_query(ctx, secret, embedding_values,
                                         params.num_clusters, params.desired_cluster_index);
    }

    std::vector<RLWECT> final_result;
    ReconstructedQuery query;
    std::vector<RLWECTEvalForm> query_eval;
    std::vector<RLWEGadgetCT> rgsw_ct;
    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_unpack(rec, "query unpacking");
            query = reconstruct_query(ctx, params, query_wire);
        }

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            query_eval.reserve(query.embedding_cts.size());
            for (const auto& lwe_ct : query.embedding_cts) {
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                query_eval.emplace_back(rlwe_ct);
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

            final_result.reserve(static_cast<size_t>(params.splits_per_cluster));
            for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
                final_result.emplace_back(ctx.rlwe_param);
            }

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

            rec.add_sample("  -> db polynomial building", db_build_time.count());
            rec.add_sample("  -> score computation", score_time.count());
            rec.add_sample("  -> RGSW masking", mask_time.count());
            rec.add_sample("  -> cross-cluster summation", sum_time.count());
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
    print_params(std::cout, params, params_source);

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
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }

    std::cout << "\n=== Per-query latency (seeded) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
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
