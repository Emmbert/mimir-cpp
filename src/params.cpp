#include "params.hpp"

#include <cmath>
#include <stdexcept>

namespace psearch {

void Params::derive_dependent_parameters() {
    if (num_clusters <= 0 || num_servers <= 0 || n <= 0) {
        throw std::invalid_argument("num_clusters, num_servers and n must be set and > 0 "
                                     "before calling derive_dependent_parameters()");
    }
    auto is_power_of_two = [](int64_t x) { return x > 0 && (x & (x - 1)) == 0; };
    if (!is_power_of_two(decomposition_base_ksk)) {
        throw std::invalid_argument("decomposition_base_ksk (" + std::to_string(decomposition_base_ksk) +
                                     ") must be a power of two -- FHE-Deck's gadget decomposition "
                                     "silently produces wrong results otherwise.");
    }
    if (!is_power_of_two(decomposition_base_prime)) {
        throw std::invalid_argument("decomposition_base_prime (" + std::to_string(decomposition_base_prime) +
                                     ") must be a power of two -- FHE-Deck's gadget decomposition "
                                     "silently produces wrong results otherwise.");
    }

    cluster_size = database_size / num_clusters; // int division, as specified
    splits_per_cluster = (cluster_size + n - 1) / n; // ceil(cluster_size / n)
    clusters_per_server = num_clusters / num_servers; // int division, as specified
}

    Params Params::make_test_params() {
    Params p;
    // Verified working via sanity_check.cpp (LWE -> RLWE -> decrypt round trip).
    p.n = 4096;//4096;
    p.q = 102445068478701569;
    p.sigma = 3.2;
    p.plaintext_modulus = 12288; //71;
    p.decomposition_base_ksk = 4; // 16
    p.decomposition_base_prime = 4; // 16
    // you settled on for the RGSW switch.

    p.database_size = 3213835; // most tests pin this locally anyway (splits_per_cluster == 1);
    p.embedding_length = 4; //4;
    p.embedding_precision = 4; //2; // signed range [-2, 1]
    p.num_clusters = 128; //4;
    p.num_servers = 10;
    p.desired_cluster_index = 2;

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
    p.num_servers = 10; // single-machine, single-threaded benchmark
    p.desired_cluster_index = 0; // arbitrary; doesn't affect timing meaningfully

    p.database_size = 3213835;

    p.derive_dependent_parameters();
    return p;
}



CryptoContext CryptoContext::from_params(const Params& params) {
    CryptoContext ctx;
    ctx.rlwe_param = std::make_shared<const FHEDeck::RLWEParam>(
        FHEDeck::RingType::negacyclic, params.n, params.q, FHEDeck::PolynomialArithmetic::ntt64);
    ctx.gadget = std::make_shared<FHEDeck::SignedDecompositionGadget>(
        params.n, params.q, params.decomposition_base_ksk);
    ctx.encoding = FHEDeck::PlaintextEncoding(
        FHEDeck::PlaintextEncodingType::full_domain, params.plaintext_modulus, params.q);
    return ctx;
}

} // namespace psearch