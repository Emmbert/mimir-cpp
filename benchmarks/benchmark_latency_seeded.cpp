// benchmark_latency_seeded_fulldb.cpp
//
// Seeded, single-threaded latency benchmark, FULL-DATABASE variant. Eval keys
// AND the query are built with seeds, wire-compressed, and reconstructed with
// NO secret key involved. The entire database is built ONCE per query and held
// resident (untimed preprocessing), so scoring streams every distinct DB
// polynomial as a real query does.
//
// This is the more precise variant: if it OOMs for a large parameter set, use
// benchmark_latency_seeded_pool.cpp instead.
//
// Everything that crosses the network is bit-packed into byte streams (see
// stream_packing.hpp) and the receiving side works ONLY on what it unpacked:
//   - eval keys: one stream (one-time),
//   - query:     one stream per component ring (client -> server),
//   - response:  one stream per component ring, s RLWE ciphertexts each
//                (server -> client).
// The results file reports the theoretical vs. actual stream sizes right
// after the parameters (see communication_cost.hpp).
//
// Run directly (NOT via ctest, which would swallow the printed table):
//   OMP_NUM_THREADS=1 ./benchmark_latency_seeded_fulldb
//   OMP_NUM_THREADS=1 ./benchmark_latency_seeded_fulldb params.json 1

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <filesystem>

#include "communication_cost.hpp"
#include "crt.hpp"
#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "params_io.hpp"
#include "seeded_distribution.hpp"
#include "seeded_eval_keys.hpp"
#include "seeded_query.hpp"
#include "stream_packing.hpp"
#include "timing.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

constexpr int kSetupWarmupRuns = 2;
constexpr int kSetupMeasuredRuns = 50;

constexpr int kQueryWarmupRuns = 1;
constexpr int kQueryMeasuredRuns = 3;

std::filesystem::path compute_output_path(const std::string& params_arg) {
    const std::string base_name = "benchmark_latency_seeded_results.txt";

    if (params_arg.empty()) {
        return std::filesystem::path(base_name);
    }

    std::filesystem::path param_path(params_arg);
    std::string stem = param_path.stem().string(); // "mimirI.json" -> "mimirI"

    std::filesystem::create_directories(stem); // no-op if it already exists
    return std::filesystem::path(stem) / base_name;
}

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

/// Untimed preprocessing. Builds the FULL database, freeing each polynomial's
/// raw_values (test-only, never read during scoring) to halve peak memory.
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

std::vector<size_t> stream_sizes(const std::vector<ByteStream>& streams) {
    std::vector<size_t> sizes;
    sizes.reserve(streams.size());
    for (const auto& s : streams) {
        sizes.push_back(s.size());
    }
    return sizes;
}

ClientPublicMaterial run_setup_and_registration(const CryptoContext& ctx, const Params& params,
                                                 ClientSecretMaterial& secret, LatencyRecorder& rec,
                                                 MeasuredStreamSizes& sizes) {
    {
        ScopedTimer t(rec, "client_setup");
        secret = generate_client_secret_material(ctx, params);
    }

    SeededClientPublicMaterial eval_wire;
    {
        ScopedTimer t(rec, "client eval key generation (seeded)");
        eval_wire = build_seeded_public_material(ctx, secret);
    }

    // --- Client: bit-pack the eval keys into ONE byte stream (this is what is sent).
    ByteStream eval_key_stream;
    {
        ScopedTimer t(rec, "client eval key stream packing");
        eval_key_stream = pack_public_material(params, eval_wire);
    }
    sizes.record_keys(eval_key_stream.size());

    // --- Server: works only on the received stream.
    ClientPublicMaterial pub;
    {
        ScopedTimer t(rec, "eval key unpacking");
        SeededClientPublicMaterial received;
        {
            ScopedTimer t_stream(rec, "eval key stream unpacking");
            received = unpack_public_material(params, eval_key_stream);
        }
        pub = reconstruct_public_material(ctx, params, received);
    }
    return pub;
}

void run_one_query(const CryptoContext& ctx, const Params& params, ClientSecretMaterial& secret,
                    const ClientPublicMaterial& pub, std::mt19937_64& rng, LatencyRecorder& rec,
                    MeasuredStreamSizes& sizes) {
    int64_t r = params.num_component_rings;

    DbEval db_eval = build_database(ctx, params, rng); // untimed

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

    // --- Client: one bit-packed stream per component ring (this is what is sent).
    std::vector<ByteStream> upload_streams; // [ring]
    {
        ScopedTimer t(rec, "client query stream packing");
        upload_streams = pack_query(params, query_wire);
    }
    std::vector<ByteStream> download_streams; // [ring]

    std::vector<std::vector<RLWECT>> final_result(static_cast<size_t>(r));
    ReconstructedQuery query;
    std::vector<std::vector<RLWECTEvalForm>> query_eval(static_cast<size_t>(r));
    std::vector<RLWEGadgetCT> rgsw_ct;
    {
        ScopedTimer t(rec, "server_processing");

        {
            ScopedTimer t_unpack(rec, "query unpacking"); // uses OpenMP internally when r>1 -- see file header
            SeededQuery received_query;
            {
                ScopedTimer t_stream(rec, "query stream unpacking");
                received_query = unpack_query(params, upload_streams);
            }
            query = reconstruct_query(ctx, params, received_query);
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

            rec.add_sample("score computation", score_time.count());
            rec.add_sample("RGSW masking", mask_time.count());
            rec.add_sample("cross-cluster summation", sum_time.count());
        }

        {
            // --- Server: one bit-packed stream per component ring, s ciphertexts each.
            ScopedTimer t_pack(rec, "response stream packing");
            download_streams = pack_response(params, final_result);
        }
    }

    sizes.record_query(stream_sizes(upload_streams), stream_sizes(download_streams));

    // --- Client: works only on the received streams.
    std::vector<std::vector<RLWECT>> received_result; // [ring][s]
    {
        ScopedTimer t(rec, "client response stream unpacking");
        received_result = unpack_response(ctx, params, download_streams);
    }

    {
        ScopedTimer t(rec, "client_decrypt");
        for (int64_t s = 0; s < params.splits_per_cluster; ++s) {
            std::vector<Vector> decrypted_per_ring;
            decrypted_per_ring.reserve(static_cast<size_t>(r));
            for (int64_t ring = 0; ring < r; ++ring) {
                decrypted_per_ring.push_back(secret.rlwe_sk->decrypt_vector(
                    received_result[static_cast<size_t>(ring)][static_cast<size_t>(s)],
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

    const CommunicationCost theoretical_cost = theoretical_communication_cost(params);
    MeasuredStreamSizes stream_sizes_measured;

    std::cout << "Warming up client setup/registration (" << kSetupWarmupRuns << " runs)...\n";
    for (int i = 0; i < kSetupWarmupRuns; ++i) {
        run_setup_and_registration(ctx, params, secret, rec, stream_sizes_measured);
    }
    rec.clear();

    std::cout << "Measuring client setup/registration (" << kSetupMeasuredRuns << " runs)...\n";
    ClientPublicMaterial pub;
    for (int i = 0; i < kSetupMeasuredRuns; ++i) {
        pub = run_setup_and_registration(ctx, params, secret, rec, stream_sizes_measured);
    }

    std::cout << "\n=== Client setup / registration latency (seeded) ===\n";
    rec.print_summary();

    LatencyRecorder query_rec;

    std::cout << "\nWarming up per-query pipeline (" << kQueryWarmupRuns << " runs)...\n";
    for (int i = 0; i < kQueryWarmupRuns; ++i) {
        run_one_query(ctx, params, secret, pub, rng, query_rec, stream_sizes_measured);
    }
    query_rec.clear();

    std::cout << "Measuring per-query pipeline (" << kQueryMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kQueryMeasuredRuns; ++i) {
        std::cout << "Iteration " << i << " " << std::flush;
        run_one_query(ctx, params, secret, pub, rng, query_rec, stream_sizes_measured);
    }

    std::cout << "\n=== Per-query latency (seeded, full database in memory) ===\n";
    query_rec.print_summary();

    std::cout << "\n";
    print_communication_cost(std::cout, params, theoretical_cost, stream_sizes_measured);

    std::ofstream out(kOutputFilePath);
    if (out) {
        print_params(out, params, params_source);
        print_communication_cost(out, params, theoretical_cost, stream_sizes_measured);
        out << "=== Client setup / registration latency (seeded) ===\n";
        rec.print_summary(out);
        out << "\n=== Per-query latency (seeded, full database in memory) ===\n";
        query_rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}