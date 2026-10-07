#include "communication_cost.hpp"

#include <iomanip>
#include <sstream>
#include <string>

#include "seeded_distribution.hpp"
#include "stream_packing.hpp"

namespace psearch {

namespace {

constexpr double kSeedBits = 8.0 * static_cast<double>(SeededUniformDistribution::kSeedBytes); // 128
constexpr double kBytesPerMiB = 1024.0 * 1024.0;

std::string format_row(const std::string& label, double theoretical, double actual) {
    std::ostringstream os;
    double overhead_pct = theoretical > 0.0 ? 100.0 * (actual - theoretical) / theoretical : 0.0;
    os << std::left << std::setw(30) << label << std::right << std::fixed << std::setprecision(1) << std::setw(16)
       << theoretical << std::setprecision(4) << std::setw(14) << theoretical / kBytesPerMiB << std::setprecision(0)
       << std::setw(16) << actual << std::setprecision(4) << std::setw(14) << actual / kBytesPerMiB
       << std::setprecision(3) << std::setw(11) << overhead_pct << " %\n";
    return os.str();
}

std::string join_streams(const std::vector<std::size_t>& sizes) {
    std::ostringstream os;
    for (std::size_t k = 0; k < sizes.size(); ++k) {
        os << (k ? ", " : "") << "ring " << k << " = " << sizes[k] << " B";
    }
    return os.str();
}

} // namespace

CommunicationCost theoretical_communication_cost(const Params& params) {
    WireShape w = wire_shape(params);
    auto bits = static_cast<double>(w.bits);
    auto n = static_cast<double>(w.n);
    auto r = static_cast<double>(w.num_component_rings);

    double key_coeffs = static_cast<double>(w.automorphism_levels * w.ksk_digits) * n +
                        2.0 * static_cast<double>(w.prime_digits) * n;
    double upload_coeffs = r * static_cast<double>(w.embedding_length) +
                           static_cast<double>(w.num_clusters * w.prime_digits);
    double download_coeffs = r * static_cast<double>(w.splits_per_cluster) * 2.0 * n;

    CommunicationCost c;
    c.keys_bytes = (kSeedBits + key_coeffs * bits) / 8.0;
    c.upload_bytes = (r * kSeedBits + upload_coeffs * bits) / 8.0;
    c.download_bytes = (download_coeffs * bits) / 8.0;
    return c;
}

void MeasuredStreamSizes::record_keys(std::size_t bytes) {
    if (keys_bytes != 0 && keys_bytes != bytes) {
        consistent = false;
    }
    keys_bytes = bytes;
}

void MeasuredStreamSizes::record_query(const std::vector<std::size_t>& upload,
                                       const std::vector<std::size_t>& download) {
    if ((!upload_stream_bytes.empty() && upload_stream_bytes != upload) ||
        (!download_stream_bytes.empty() && download_stream_bytes != download)) {
        consistent = false;
    }
    upload_stream_bytes = upload;
    download_stream_bytes = download;
}

CommunicationCost MeasuredStreamSizes::as_cost() const {
    CommunicationCost c;
    c.keys_bytes = static_cast<double>(keys_bytes);
    for (std::size_t b : upload_stream_bytes) {
        c.upload_bytes += static_cast<double>(b);
    }
    for (std::size_t b : download_stream_bytes) {
        c.download_bytes += static_cast<double>(b);
    }
    return c;
}

void print_communication_cost(std::ostream& os, const Params& params, const CommunicationCost& theoretical,
                              const MeasuredStreamSizes& actual) {
    WireShape w = wire_shape(params);
    CommunicationCost act = actual.as_cost();

    std::ios_base::fmtflags saved_flags = os.flags();
    std::streamsize saved_precision = os.precision();

    os << "=== Communication cost (bit-packed streams) ===\n";
    os << "bits per coefficient     = " << w.bits << " (ceil(log2 q)), seeds = 128 bits\n";
    os << "digits (ksk / prime)     = " << w.ksk_digits << " / " << w.prime_digits
       << ", automorphism levels = " << w.automorphism_levels << "\n";
    os << "streams per query        = " << w.num_component_rings << " up + " << w.num_component_rings
       << " down (one per component ring)\n";
    os << std::left << std::setw(30) << "item" << std::right << std::setw(16) << "theoretical(B)" << std::setw(14)
       << "theor.(MiB)" << std::setw(16) << "actual(B)" << std::setw(14) << "actual(MiB)" << std::setw(13)
       << "overhead" << "\n";
    os << format_row("one-time eval keys", theoretical.keys_bytes, act.keys_bytes);
    os << format_row("upload per query", theoretical.upload_bytes, act.upload_bytes);
    os << format_row("download per query", theoretical.download_bytes, act.download_bytes);
    os << format_row("total per query (up + down)", theoretical.per_query_bytes(), act.per_query_bytes());
    os << "actual upload streams    : " << join_streams(actual.upload_stream_bytes) << "\n";
    os << "actual download streams  : " << join_streams(actual.download_stream_bytes) << "\n";
    if (!actual.consistent) {
        os << "WARNING: measured stream sizes differed between runs -- the numbers above are from the last run.\n";
    }
    os << "\n";

    os.flags(saved_flags);
    os.precision(saved_precision);
}

} // namespace psearch
