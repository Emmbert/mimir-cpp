#include "params.hpp"

#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <cstdlib>       // std::getenv
#include "crt.hpp"        // crt_combined_modulus
#include "params_io.hpp" // load_params_from_json

namespace psearch {

void Params::derive_dependent_parameters() {
    if (num_clusters <= 0 || num_servers <= 0 || n <= 0) {
        throw std::invalid_argument("num_clusters, num_servers and n must be set and > 0 "
                                     "before calling derive_dependent_parameters()");
    }
    if (q >= (uint64_t(1) << 62)) {
        throw std::invalid_argument("ciphertext modulus is too large for NTT. It has to use < 63 bits");
    }

    auto is_power_of_two = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
    /*if (!is_power_of_two(decomposition_base_ksk)) {
        throw std::invalid_argument("decomposition_base_ksk (" + std::to_string(decomposition_base_ksk) +
                                     ") must be a power of two -- FHE-Deck's gadget decomposition "
                                     "silently produces wrong results otherwise.");
    }
    if (!is_power_of_two(decomposition_base_prime)) {
        throw std::invalid_argument("decomposition_base_prime (" + std::to_string(decomposition_base_prime) +
                                     ") must be a power of two -- FHE-Deck's gadget decomposition "
                                     "silently produces wrong results otherwise.");
    }*/

    cluster_size = (database_size + num_clusters - 1) / num_clusters; // ceil(database_size / num_clusters)
    splits_per_cluster = (cluster_size + n - 1) / n; // ceil(cluster_size / n)
    clusters_per_server = (num_clusters + num_servers - 1) / num_servers; // ceil()

    // ---- CRT validation ---------------------------------------------------
    if (num_component_rings != 1 && num_component_rings != 2) {
        throw std::invalid_argument("num_component_rings must be 1 (no CRT) or 2 (two-component-ring CRT), got " +
                                     std::to_string(num_component_rings));
    }
    if (num_component_rings == 2) {
        if (comp_ring_modulus < 2) {
            throw std::invalid_argument("comp_ring_modulus must be set (>= 2) when num_component_rings == 2, got " +
                                         std::to_string(comp_ring_modulus));
        }
        combined_component_ring_modulus = crt_combined_modulus(comp_ring_modulus);
        if (combined_component_ring_modulus < plaintext_modulus) {
            throw std::invalid_argument(
                "comp_ring_modulus (" + std::to_string(comp_ring_modulus) + ") gives a combined CRT modulus of " +
                std::to_string(combined_component_ring_modulus) +
                " (= comp_ring_modulus*(comp_ring_modulus-1)), which is smaller than "
                "plaintext_modulus (" + std::to_string(plaintext_modulus) +
                "). plaintext_modulus is the required lower bound on the message space size -- CRT is just an "
                "implementation detail of how that space is represented, so the combined modulus must still meet "
                "it, exactly as the single-ring case would need plaintext_modulus itself to be at least this large.");
        }
    } else {
        combined_component_ring_modulus = 0; // not meaningful outside CRT
    }
}

    Params Params::make_test_params() {
    // Opt-in: if MIMIR_TEST_PARAMS_FILE is set, load real parameters from
    // that file instead of the small hardcoded ones below. This lets the
    // EXACT SAME test suite run against a real parameter set -- no test file
    // needs to change, nothing needs to be typed by hand. OFF by default, so
    // a plain `ctest` run stays fast.
    //
    // Some tests loop over every cluster, single-threaded (e.g.
    // test_full_scoring_with_cluster_selection.cpp, test_full_scoring_with_splits.cpp)
    // -- against a real file (num_clusters in the hundreds), those can take
    // tens of seconds instead of milliseconds. Prefer running specific tests
    // rather than the whole suite when this is set, e.g.:
    //   MIMIR_TEST_PARAMS_FILE=path/to/params.json ctest -R FullScoringWithSplitsParallel
    //   MIMIR_TEST_PARAMS_FILE=path/to/params.json ./test_full_scoring_with_splits_parallel
    if (const char* env_path = std::getenv("MIMIR_TEST_PARAMS_FILE")) {
        std::string path(env_path);
        if (!path.empty()) {
            std::cout << "MIMIR_TEST_PARAMS_FILE set -- loading test params from " << path << "\n";
            // num_servers/desired_cluster_index don't affect what any
            // correctness test actually checks -- 1 and 0 are safe,
            // always-valid defaults regardless of the file's num_clusters.
            return load_params_from_json(path, /*num_servers=*/1, /*desired_cluster_index=*/0);
        }
    }

    Params p;
    // Verified working via sanity_check.cpp (LWE -> RLWE -> decrypt round trip).
    p.n = 2048;
    p.q = 281474976694273;
    p.sigma = 3.2;
    p.plaintext_modulus = 71;
    p.decomposition_base_ksk = 4; // 16
    p.decomposition_base_prime = 8; // 16
    // you settled on for the RGSW switch.

    p.database_size = 321383; // most tests pin this locally anyway (splits_per_cluster == 1);
    p.embedding_length = 4; //4;
    p.embedding_precision = 2; //2; // signed range [-2, 1]
    p.num_clusters = 24; //4;
    p.num_servers = 10;
    p.desired_cluster_index = 2;

    p.derive_dependent_parameters();
    return p;
}

    Params Params::make_test_params_component_rings() {
    // Same MIMIR_TEST_PARAMS_FILE opt-in as make_test_params() -- "as
    // usual". A file with r_NUM_COMP_RINGS == 1 loads fine here (it's
    // valid Params), but CRT-specific tests should skip rather than run
    // against it -- see test_crt.cpp for the skip pattern.
    if (const char* env_path = std::getenv("MIMIR_TEST_PARAMS_FILE")) {
        std::string path(env_path);
        if (!path.empty()) {
            std::cout << "MIMIR_TEST_PARAMS_FILE set -- loading test params from " << path << "\n";
            return load_params_from_json(path, /*num_servers=*/1, /*desired_cluster_index=*/0);
        }
    }

    // Mirrors make_test_params()'s n/q/sigma/decomposition bases/
    // database_size/embedding fields exactly, so a non-CRT and CRT test
    // can be directly compared on otherwise-identical parameters.
    Params p;
    p.n = 2048;
    p.q = 281474976694273;
    p.sigma = 3.2;
    p.plaintext_modulus = 71; // same lower bound as make_test_params() -- unchanged meaning
    p.decomposition_base_ksk = 4;
    p.decomposition_base_prime = 8;

    p.database_size = 321383;
    p.embedding_length = 4;
    p.embedding_precision = 2;
    p.num_clusters = 24;
    p.num_servers = 10;
    p.desired_cluster_index = 2;

    p.num_component_rings = 2;
    p.comp_ring_modulus = 11; // p1=11, p2=10, combined=110 -- comfortably >= plaintext_modulus (71)

    p.derive_dependent_parameters();
    return p;
}

    constexpr const char* Params::kTestDatabaseFilePath;

    Params Params::make_test_database_params() {
        Params p;
        p.n = 4096;
        p.q = 281474976694273; // ~2^48, same modulus used throughout earlier examples
        p.sigma = 3.2;
        p.plaintext_modulus = 103; // generous margin: dot_product_can_overflow's
        // worst case here is 192*8*8=12288, well under 65536/2
        p.decomposition_base_ksk = 4;
        p.decomposition_base_prime = 8;

        p.database_size = 5100;
        p.embedding_length = 10;
        p.embedding_precision = 2;
        p.num_clusters = 2;
        p.num_servers = 1;
        p.desired_cluster_index = 0;

        p.derive_dependent_parameters();
        return p;
    }

    Params Params::make_test_database_params_with_splits() {
        Params p = make_test_database_params();
        p.n = 2048; // splits_per_cluster = ceil(2550/2048) = 2 per cluster --
        // first split fully real, second split 502 real + 1546
        // zero-padded, exercising the padding path against real data.
        p.derive_dependent_parameters();
        return p;
    }


    Params Params::make_benchmark_params() {
    Params p;
    p.n = 2048;
    p.q = 281474976694273;
    p.sigma = 3.2;
    // embedding_precision=4 -> signed range [-8, 7] -> worst-case dot product
    // magnitude = embedding_length * 8^2 = 192 * 64 = 12288. Needs
    // plaintext_modulus > 2*12288 = 24576 with margin; 100003 gives ~4x that.
    p.plaintext_modulus = 100003;
    p.decomposition_base_ksk = 16;
    p.decomposition_base_prime = 8;

    p.embedding_length = 64;
    p.embedding_precision = 4;
    p.num_clusters = 24;
    p.num_servers = 10;
    p.desired_cluster_index = 0;

    p.database_size = 3213835;

    p.derive_dependent_parameters();
    return p;
}



    CryptoContext CryptoContext::from_params(const Params& params) {
    CryptoContext ctx;
    ctx.rlwe_param = std::make_shared<const FHEDeck::RLWEParam>(
        FHEDeck::RingType::negacyclic, params.n, params.q, FHEDeck::PolynomialArithmetic::ntt64);
    ctx.gadget_ksk = std::make_shared<FHEDeck::SignedDecompositionGadget>(
        params.n, params.q, params.decomposition_base_ksk);
    ctx.gadget_rgsw = std::make_shared<FHEDeck::SignedDecompositionGadget>(
        params.n, params.q, params.decomposition_base_prime);
    ctx.encoding = FHEDeck::PlaintextEncoding(
        FHEDeck::PlaintextEncodingType::full_domain, params.plaintext_modulus, params.q);

    if (params.num_component_rings == 1) {
        ctx.component_encodings = {ctx.encoding};
    } else { // == 2, validated by Params::derive_dependent_parameters()
        int64_t p1 = params.comp_ring_modulus;
        int64_t p2 = params.comp_ring_modulus - 1;
        ctx.component_encodings = {
            FHEDeck::PlaintextEncoding(FHEDeck::PlaintextEncodingType::full_domain, p1, params.q),
            FHEDeck::PlaintextEncoding(FHEDeck::PlaintextEncodingType::full_domain, p2, params.q),
        };
    }
    return ctx;
}


} // namespace psearch