// benchmark_latency_fulldb.cpp
//
// Single-machine, single-threaded latency benchmark, FULL-DATABASE variant.
// The entire database (num_clusters x splits x l x r eval-form polynomials)
// is built ONCE per query and held resident, so scoring streams every
// distinct DB polynomial exactly as a real query does. Database construction
// is preprocessing (server setup) and is NEVER timed.
//
// This is the more precise variant: if it OOMs for a large parameter set,
// use benchmark_latency_pool.cpp instead (a small fixed pool bounds memory
// but makes scoring cache-unrealistically fast).
//
// Run directly (NOT via ctest, which would swallow the printed table):
//   ./benchmark_latency_fulldb
//   ./benchmark_latency_fulldb params.json 1

// TODO embedding sampling should be moved out of the query generation time (because the rejection sampling takes longer
// TODO than the real embedding generation

#include <chrono>
#include <fstream>
#include <iostream>
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

const char* kOutputFilePath = "benchmark_latency_results.txt";

// Full database, laid out [c][s][ring][j] so db_eval[c][s][ring] is directly
// the length-l vector compute_split_score wants -- no per-j gather needed.
using DbEval = std::vector<std::vector<std::vector<std::vector<DatabasePolynomialEvalForm>>>>; // [c][s][ring][j]

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

/// Preprocessing -- deliberately NOT wrapped in any ScopedTimer. Builds the
/// FULL database, freeing each polynomial's raw_values (test-only, never read
/// during scoring) to halve peak memory.
DbEval build_database(const CryptoContext& ctx, const Params& params, std::mt19937_64& rng) {
    int64_t r = params.num_component_rings;
    DbEval db_eval(static_cast<size_t>(params.num_clusters));
    for (int64_t c = 0; c < params.num_clusters; ++c) {
        db_eval[static_cast<size_t>(c)].resize(static_cast<size_t>(params.splits_per_cluster));
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            auto& entry = db_eval[static_cast<size_t>(c)][static_cast<size_t>(s)]; // [ring][j]
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
    }
    return db_eval;
}

void run_setup_and_registration(const CryptoContext& ctx, const Params& params, LatencyRecorder& rec) {
    ClientSecretMaterial secret;
    {
        ScopedTimer t(rec, "client_setup");
        secret = generate_client_secret_material(ctx, params);
        [[maybe_unused]] ClientPublicMaterial pub = generate_client_public_material(ctx, secret);
    }
}

void run_one_query(const CryptoContext& ctx, const Params& params, const ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec) {
    int64_t r = params.num_component_rings;
    int64_t combined_modulus = (r == 1) ? params.plaintext_modulus : params.combined_component_ring_modulus;

    // Untimed preprocessing.
    DbEval db_eval = build_database(ctx, params, rng);

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
    std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r));
    std::vector<RLWEGadgetCT> rgsw_ct;
    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_switch_rlwe(rec, "RLWE ciphertext switching");
            for (int64_t ring = 0; ring < r; ++ring) {
                query_eval[static_cast<size_t>(ring)].reserve(static_cast<size_t>(params.embedding_length));
                for (const auto& lwe_ct : embedding_lwe[static_cast<size_t>(ring)]) {
                    RLWECT rlwe_ct(ctx.rlwe_param);
                    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                    query_eval[static_cast<size_t>(ring)].emplace_back(rlwe_ct);
                }
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
                    for (int64_t ring = 0; ring < r; ++ring) {
                        // Direct const ref into the full DB -- raw_values were freed at build time.
                        const std::vector<DatabasePolynomialEvalForm>& db_for_ring =
                            db_eval[static_cast<size_t>(c)][static_cast<size_t>(s)][static_cast<size_t>(ring)];

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

    std::mt19937_64 rng(std::random_device{}());

    LatencyRecorder rec;

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
        std::cout << "Iteration " << i << " " << std::flush;
        run_one_query(ctx, params, secret, pub, rng, query_rec);
    }

    std::cout << "\n=== Per-query latency (full database in memory) ===\n";
    query_rec.print_summary();

    std::ofstream out(kOutputFilePath);
    if (out) {
        print_params(out, params, params_source);
        out << "=== Client setup / registration latency ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (full database in memory) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}
