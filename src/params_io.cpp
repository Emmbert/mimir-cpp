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

int64_t round_up_to_power_of_two(int64_t x) {
    if (x <= 1) return 1;
    int64_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/// Reads a gadget base field, rounding up to the nearest power of two if
/// necessary (with a warning) -- see params_io.hpp for why this is needed.
/// TODO: if you'd rather fail loudly instead of approximating, replace the
/// rounding branch below with:
///   throw std::runtime_error("load_params_from_json: " + key + " (" +
///       std::to_string(base) + ") is not a power of two, and this loader "
///       "is configured to reject that rather than approximate.");
int64_t read_gadget_base(const json& j, const std::string& key) {
    int64_t base = read_int_required(j, key);
    int64_t rounded = round_up_to_power_of_two(base);
    if (rounded != base) {
        std::cerr << "WARNING: load_params_from_json: " << key << " = " << base
                  << " is not a power of two (fhe-deck-core's gadget decomposition requires "
                  << "one). Rounding UP to " << rounded << ". This means the loaded parameters "
                  << "no longer exactly match this file's own DFR / communication-cost "
                  << "predictions, which assumed base " << base << ".\n";
    }
    return rounded;
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

    // This codebase doesn't implement CRT decomposition (multiple
    // computation rings) yet -- a file that requires it can't be faithfully
    // represented.
    int64_t num_comp_rings = read_int_required(j, "r_NUM_COMP_RINGS");
    if (num_comp_rings != 1) {
        throw std::runtime_error("load_params_from_json: " + path + " requires r_NUM_COMP_RINGS = " +
                                  std::to_string(num_comp_rings) +
                                  ", but this codebase only supports a single computation ring (CRT "
                                  "decomposition isn't implemented -- see future/NOTES.md).");
    }

    Params p;
    p.n = read_int_required(j, "n_LATTICE_DIMENSION");
    p.q = read_int_required(j, "q_CIPHERTEXT_MODULUS");
    p.sigma = read_number_flexible(j, "sigma_STANDARD_DEV");
    p.plaintext_modulus = read_int_required(j, "PLAINTEXT_MODULUS");

    p.decomposition_base_ksk = read_gadget_base(j, "B_BASE_DIGIT_KS");
    p.decomposition_base_prime = read_gadget_base(j, "B_BASE_DIGIT_PRIME");

    p.database_size = read_int_required(j, "N_NUM_DOCS");
    p.embedding_length = read_int_required(j, "l_EMBEDDING_DIM");
    p.embedding_precision = read_int_required(j, "EMBEDDING_PRECISION");
    p.num_clusters = read_int_required(j, "C_NUM_CLUSTERS");

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

    p.derive_dependent_parameters(); // also validates both gadget bases are powers of two
    return p;
}

Params load_benchmark_params_from_args(int argc, char** argv) {
    if (argc >= 3) {
        std::string path = argv[1];
        int64_t num_servers = std::stoll(argv[2]);
        int64_t desired_cluster_index = (argc >= 4) ? std::stoll(argv[3]) : 0;
        return load_params_from_json(path, num_servers, desired_cluster_index);
    }
    std::cout << "No parameter file given (usage: " << (argc >= 1 ? argv[0] : "<binary>")
              << " <params.json> <num_servers> [desired_cluster_index]); "
              << "falling back to Params::make_benchmark_params().\n";
    return Params::make_benchmark_params();
}

} // namespace psearch