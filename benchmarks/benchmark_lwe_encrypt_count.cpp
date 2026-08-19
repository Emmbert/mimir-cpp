// benchmark_lwe_encrypt_count.cpp
//
// Sanity-check for the query-generation cost. Times
//
//     N = embedding_length + num_clusters * ceil(log_64(q))
//
// sequential LWE encryptions (encode_and_encrypt) -- approximately what
// building one query costs: embedding_length embedding LWE encryptions, plus
// one gadget_encrypt per selector cluster, where each gadget_encrypt is worth
// roughly ceil(log_64(q)) LWE encryptions (one per base-64 gadget digit).
//
// Compare the total against your "client_query_gen (seeded)" number (~470 ms)
// to check whether query generation is fully explained by the count and cost
// of individual LWE encryptions.
//
// Commit-agnostic: uses only APIs and fields identical across the pre-CRT and
// CRT commits, so the SAME file compiles and runs unchanged under both.
//
// Build (CMakeLists.txt, near the other benchmark targets):
//   add_executable(benchmark_lwe_encrypt_count benchmarks/benchmark_lwe_encrypt_count.cpp)
//   target_link_libraries(benchmark_lwe_encrypt_count PRIVATE mimir_core)
//
// Run:
//   OMP_NUM_THREADS=1 ./benchmark_lwe_encrypt_count ../parameter_files/<the same file>.json

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"
#include "params_io.hpp"
#include "timing.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {
constexpr int kWarmupRuns = 2;
constexpr int kMeasuredRuns = 10;
const char* kOutputFilePath = "benchmark_lwe_encrypt_count_results.txt";
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <params.json>\n";
        return 1;
    }
    const std::string params_path = argv[1];
    Params params = load_params_from_json(params_path, /*num_servers=*/1, /*desired_cluster_index=*/0);
    CryptoContext ctx = CryptoContext::from_params(params);
    print_params(std::cout, params, params_path);

    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);

    // gadget_digits = ceil(log_64(q)): how many base-64 gadget digits, i.e. how
    // many LWE encryptions one gadget_encrypt is worth (approximately).
    const int64_t gadget_digits =
        static_cast<int64_t>(std::ceil(std::log(static_cast<double>(params.q)) / std::log(64.0)));

    const int64_t N = params.embedding_length + params.num_clusters * gadget_digits;

    std::cout << "\ngadget_digits = ceil(log_64(q)) = " << gadget_digits << "\n";
    std::cout << "N = embedding_length + num_clusters * gadget_digits = " << params.embedding_length << " + "
              << params.num_clusters << " * " << gadget_digits << " = " << N << " LWE encryptions\n";
    std::cout << "(each encryption samples an n = " << params.n << " uniform a-vector mod q)\n";

    auto run_once = [&](LatencyRecorder& rec) {
        std::vector<LWECT> cts;
        cts.reserve(static_cast<size_t>(N));
        {
            ScopedTimer t(rec, "N LWE encryptions");
            for (int64_t i = 0; i < N; ++i) {
                cts.push_back(secret.lwe_sk->encode_and_encrypt(static_cast<int64_t>(1), ctx.encoding));
            }
        }
        volatile size_t sink = cts.size();
        (void)sink;
    };

    LatencyRecorder rec;
    std::cout << "\nWarming up (" << kWarmupRuns << " runs)...\n";
    for (int i = 0; i < kWarmupRuns; ++i) run_once(rec);
    rec.clear();

    std::cout << "Measuring (" << kMeasuredRuns << " runs)...\n";
    for (int i = 0; i < kMeasuredRuns; ++i) {
        std::cout << "Iteration " << i << " " << std::flush;
        run_once(rec);
    }
    std::cout << "\n";

    const double med = rec.median_ms("N LWE encryptions");
    std::cout << "\n=== " << N << " sequential LWE encryptions ===\n";
    rec.print_summary();
    std::cout << "\nPer-encryption (median): " << (med / static_cast<double>(N)) << " ms/encryption"
              << "  (n = " << params.n << " coefficients each)\n";

    std::ofstream out(kOutputFilePath);
    if (out) {
        print_params(out, params, params_path);
        out << "gadget_digits = ceil(log_64(q)) = " << gadget_digits << "\n";
        out << "N = " << N << " LWE encryptions\n";
        rec.print_summary(out);
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }
    return 0;
}
