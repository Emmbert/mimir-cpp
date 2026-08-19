// benchmark_switch_isolated.cpp
//
// Standalone micro-benchmark that isolates and times ONLY three things, with
// NO database, NO scoring, NO decryption, and nothing else running that could
// perturb them:
//
//   (1) client-side query encryption -- l embedding LWE ciphertexts
//       (encode_and_encrypt) + num_clusters gadget-encrypted selector
//       ciphertexts (gadget_encrypt);
//   (2) RLWE ciphertext switching -- LWE->RLWE key switch + conversion to
//       eval form, byte-for-byte the same operations the server's per-query
//       loop performs;
//   (3) RGSW ciphertext switching -- LWE'->RGSW key switch for the selector.
//
// These numbers are directly comparable to the "client_query_gen (seeded)",
// "RLWE ciphertext switching" and "RGSW ciphertext switching" lines of
// benchmark_latency_seeded -- but measured in isolation.
//
// IMPORTANT: this file uses ONLY APIs and Params/CryptoContext fields that are
// identical across the pre-CRT commit (60a11be) and the CRT commit (7fac9a3),
// so the SAME file compiles and runs unchanged under BOTH. Build it under each
// commit and compare -- that tells you whether the *switch operation itself*
// differs, independent of the seeded reconstruction path.
//
// Plaintext messages are fixed: encryption and key-switching cost is
// independent of the plaintext value, so no random sampling or CRT split is
// needed here (and avoiding them keeps the file commit-agnostic).
//
// Build (add to CMakeLists.txt, near the other benchmark targets):
//   add_executable(benchmark_switch_isolated benchmarks/benchmark_switch_isolated.cpp)
//   target_link_libraries(benchmark_switch_isolated PRIVATE mimir_core)
//
// Run (single-threaded, matching the plain seeded benchmark's linkage):
//   OMP_NUM_THREADS=1 ./benchmark_switch_isolated ../parameter_files/<the same file>.json

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

const char* kOutputFilePath = "benchmark_switch_isolated_results.txt";

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

    // ---- One-time setup: keys built once and reused for every iteration,
    // exactly like a real client session (matches benchmark_latency_seeded,
    // where setup/registration is measured separately from per-query work). --
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

    const int64_t l = params.embedding_length;
    const int64_t C = params.num_clusters;
    const int64_t sel_idx = params.desired_cluster_index;

    // One full query's worth of encryption + both switches. Fresh objects each
    // call, so every iteration operates on freshly-written ciphertexts (mirrors
    // a real per-query pipeline rather than reusing cache-hot inputs).
    auto run_once = [&](LatencyRecorder& rec) {
        // (1) Client-side encryption. Fixed messages; cost is value-independent.
        std::vector<LWECT> embedding_lwe;
        std::vector<LWEGadgetCT> selector_gadget;
        {
            ScopedTimer t(rec, "client encryption (l LWE + C gadget)");
            embedding_lwe.reserve(static_cast<size_t>(l));
            for (int64_t j = 0; j < l; ++j) {
                embedding_lwe.push_back(secret.lwe_sk->encode_and_encrypt(static_cast<int64_t>(1), ctx.encoding));
            }
            selector_gadget.reserve(static_cast<size_t>(C));
            for (int64_t c = 0; c < C; ++c) {
                int64_t bit = (c == sel_idx) ? 1 : 0;
                selector_gadget.push_back(secret.lwe_gadget_sk->gadget_encrypt(bit));
            }
        }

        // (2) RLWE switching -- LWE->RLWE key switch, then build eval form.
        // This is EXACTLY what the server's "RLWE ciphertext switching" loop
        // does (the emplace into a vector<RLWECTEvalForm> performs the NTT to
        // eval form, so it is included here too, to match).
        std::vector<RLWECTEvalForm> query_eval;
        {
            ScopedTimer t(rec, "RLWE ciphertext switching");
            query_eval.reserve(static_cast<size_t>(l));
            for (const auto& lwe_ct : embedding_lwe) {
                RLWECT rlwe_ct(ctx.rlwe_param);
                pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
                query_eval.emplace_back(rlwe_ct); // constructs eval form (NTT), as in the real loop
            }
        }

        // (3) RGSW switching -- LWE'->RGSW key switch for each selector digit-set.
        std::vector<RLWEGadgetCT> rgsw_ct;
        {
            ScopedTimer t(rec, "RGSW ciphertext switching");
            rgsw_ct.reserve(static_cast<size_t>(C));
            for (const auto& gadget_ct : selector_gadget) {
                rgsw_ct.push_back(pub.lwe_to_rgsw_ksk->lwe_to_rlwe_key_switch(gadget_ct));
            }
        }

        // Prevent the optimizer from discarding the work above.
        volatile size_t sink = query_eval.size() + rgsw_ct.size() + embedding_lwe.size();
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

    std::cout << "\n=== Isolated encryption / switching latency ===\n";
    rec.print_summary();

    auto per_op = [&](const char* stage, int64_t ops) {
        const double med = rec.median_ms(stage);
        std::cout << "  " << stage << ": " << med << " ms median, "
                  << (med / static_cast<double>(ops)) << " ms/op over " << ops << " ops\n";
    };
    std::cout << "\nPer-operation (median):\n";
    per_op("client encryption (l LWE + C gadget)", l + C);
    per_op("RLWE ciphertext switching", l);
    per_op("RGSW ciphertext switching", C);

    std::ofstream out(kOutputFilePath);
    if (out) {
        print_params(out, params, params_path);
        out << "=== Isolated encryption / switching latency ===\n";
        rec.print_summary(out);
        std::cout << "\nResults written to " << kOutputFilePath << "\n";
    } else {
        std::cerr << "\nWARNING: could not open " << kOutputFilePath << " for writing.\n";
    }

    return 0;
}
