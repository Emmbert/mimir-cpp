#include "params_io.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace psearch {

namespace {

using json = nlohmann::json;

/// Some fields in these files are JSON numbers, others are JSON strings
/// holding a number (e.g. "9.1e-13", "3.7e+09") -- this reads either.
double read_number_flexible(const json& j, const std::string& key) {
    if (!j.contains(key)) {
        throw std::runtime_error("load_params_from_json: missing required field \"" + key + "\"");
    }
    const json& v = j.at(key);
    if (v.is_string()) {
        return std::stod(v.get<std::string>());
    }
    return v.get<double>();
}

int64_t read_int_required(const json& j, const std::string& key) {
    if (!j.contains(key)) {
        throw std::runtime_error("load_params_from_json: missing required field \"" + key + "\"");
    }
    const json& v = j.at(key);
    if (v.is_string()) {
        return static_cast<int64_t>(std::stoll(v.get<std::string>()));
    }
    return v.get<int64_t>();
}

} // namespace

Params load_params_from_json(const std::string& path, int64_t num_servers, int64_t desired_cluster_index) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("load_params_from_json: could not open file: " + path);
    }

    json j;
    try {
        in >> j;
    } catch (const json::parse_error& e) {
        throw std::runtime_error("load_params_from_json: malformed JSON in " + path + ": " + e.what());
    }

    // This codebase supports 1 (no CRT) or 2 (two-component-ring CRT)
    // computation rings -- see Params::num_component_rings/comp_ring_modulus
    // and crt.hpp for what "two component rings" actually means here.
    int64_t num_comp_rings = read_int_required(j, "r_NUM_COMP_RINGS");
    if (num_comp_rings != 1 && num_comp_rings != 2) {
        throw std::runtime_error("load_params_from_json: " + path + " requires r_NUM_COMP_RINGS = " +
                                  std::to_string(num_comp_rings) +
                                  ", but this codebase only supports 1 (no CRT) or 2 (two-component-ring CRT).");
    }

    Params p;
    p.n = read_int_required(j, "n_LATTICE_DIMENSION");
    p.q = read_int_required(j, "q_CIPHERTEXT_MODULUS");
    p.sigma = read_number_flexible(j, "sigma_STANDARD_DEV");
    p.plaintext_modulus = read_int_required(j, "PLAINTEXT_MODULUS");

    p.decomposition_base_ksk = read_int_required(j, "B_BASE_DIGIT_KS");
    p.decomposition_base_prime = read_int_required(j, "B_BASE_DIGIT_PRIME");

    p.database_size = read_int_required(j, "N_NUM_DOCS");
    p.embedding_length = read_int_required(j, "l_EMBEDDING_DIM");
    p.embedding_precision = read_int_required(j, "EMBEDDING_PRECISION");
    p.num_clusters = read_int_required(j, "C_NUM_CLUSTERS");

    p.num_component_rings = num_comp_rings;
    if (num_comp_rings == 2) {
        // p1 = pr_COMP_RING_MODULUS; p2 is always p1 - 1 (consecutive
        // integers, always coprime) -- see crt.hpp. The combined modulus
        // p1*(p1-1) is validated against plaintext_modulus (the required
        // lower bound, unchanged in meaning by CRT) inside
        // derive_dependent_parameters() below, not here -- so that same
        // validation applies uniformly to hand-written Params too, not
        // just ones loaded from JSON.
        p.comp_ring_modulus = read_int_required(j, "pr_COMP_RING_MODULUS");
    }

    // Not present in these files -- deployment/query choices, supplied by
    // the caller rather than the parameter-search tool.
    p.num_servers = num_servers;
    p.desired_cluster_index = desired_cluster_index;

    // Informational only: print the file's own predicted DFR for reference,
    // so it can be compared against your own measured failure rate.
    try {
        double dfr = read_number_flexible(j, "DFR");
        std::cout << "Loaded params from " << path << " (file's predicted DFR: " << dfr << ")\n";
    } catch (const std::exception&) {
        std::cout << "Loaded params from " << path << "\n";
    }

    p.derive_dependent_parameters(); // also validates gadget bases and CRT parameters
    return p;
}

    Params load_benchmark_params_from_args(int argc, char** argv, bool distributed,
                                            std::string* out_source_description) {
    if (argc >= 2) {
        std::string path = argv[1];
        int64_t num_servers = 1;
        int64_t desired_cluster_index = 0;

        if (distributed) {
            // <params.json> [num_servers] [desired_cluster_index]
            if (argc >= 3) {
                num_servers = std::stoll(argv[2]);
            }
            if (argc >= 4) {
                desired_cluster_index = std::stoll(argv[3]);
            }
        } else {
            // <params.json> [desired_cluster_index] -- num_servers is
            // meaningless for a single-machine benchmark, so it isn't a
            // positional argument at all here; argv[2] IS
            // desired_cluster_index directly, never skipped/ignored.
            if (argc >= 3) {
                desired_cluster_index = std::stoll(argv[2]);
            }
        }

        if (out_source_description) {
            *out_source_description = path;
        }
        return load_params_from_json(path, num_servers, desired_cluster_index);
    }
    std::cout << "No parameter file given (usage: " << (argc >= 1 ? argv[0] : "<binary>")
              << (distributed ? " <params.json> [num_servers] [desired_cluster_index]"
                               : " <params.json> [desired_cluster_index]")
              << "); falling back to Params::make_benchmark_params().\n";
    if (out_source_description) {
        *out_source_description = "Params::make_benchmark_params() (built-in defaults)";
    }
    return Params::make_benchmark_params();
}


void print_params(std::ostream& os, const Params& params, const std::string& source_description) {
    os << "=== Parameters (source: " << source_description << ") ===\n";
    os << "n                        = " << params.n << "\n";
    os << "q                        = " << params.q << "\n";
    os << "sigma                    = " << params.sigma << "\n";
    os << "plaintext_modulus        = " << params.plaintext_modulus << "\n";
    os << "decomposition_base_ksk   = " << params.decomposition_base_ksk << "\n";
    os << "decomposition_base_prime = " << params.decomposition_base_prime << "\n";
    os << "database_size            = " << params.database_size << "\n";
    os << "embedding_length         = " << params.embedding_length << "\n";
    os << "embedding_precision      = " << params.embedding_precision << "\n";
    os << "num_clusters             = " << params.num_clusters << "\n";
    os << "cluster_size             = " << params.cluster_size << "\n";
    os << "splits_per_cluster       = " << params.splits_per_cluster << "\n";
    os << "num_servers              = " << params.num_servers << "\n";
    os << "clusters_per_server      = " << params.clusters_per_server << "\n";
    os << "desired_cluster_index    = " << params.desired_cluster_index << "\n";
    os << "num_component_rings      = " << params.num_component_rings << "\n";
    if (params.num_component_rings == 2) {
        os << "comp_ring_modulus        = " << params.comp_ring_modulus << " (p2 = "
           << (params.comp_ring_modulus - 1) << ")\n";
    }
    os << "\n";
}

} // namespace psearch