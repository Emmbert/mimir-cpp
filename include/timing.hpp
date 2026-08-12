#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace psearch {

/// Collects repeated latency samples for named stages ("client_setup",
/// "client_registration", "client_query_gen", "server_processing",
/// "client_decrypt", ...) across many benchmark runs, and reports
/// mean/stddev/median/min/max at the end.
///
/// Usage:
///   LatencyRecorder rec;
///   {
///       ScopedTimer t(rec, "client_query_gen");
///       // ... do the work ...
///   } // duration recorded automatically on scope exit
///   rec.print_summary();
class LatencyRecorder {
public:
    void add_sample(const std::string& stage, double milliseconds) {
        samples_[stage].push_back(milliseconds);
    }

    double mean_ms(const std::string& stage) const {
        const auto& v = samples_.at(stage);
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    }

    double median_ms(const std::string& stage) const {
        auto v = samples_.at(stage); // copy, so we can sort without mutating stored samples
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t mid = v.size() / 2;
        if (v.size() % 2 == 0) return (v[mid - 1] + v[mid]) / 2.0;
        return v[mid];
    }

    double stddev_ms(const std::string& stage) const {
        const auto& v = samples_.at(stage);
        if (v.size() < 2) return 0.0;
        double m = mean_ms(stage);
        double sq_sum = 0.0;
        for (double x : v) sq_sum += (x - m) * (x - m);
        return std::sqrt(sq_sum / static_cast<double>(v.size() - 1));
    }

    double min_ms(const std::string& stage) const {
        const auto& v = samples_.at(stage);
        return v.empty() ? 0.0 : *std::min_element(v.begin(), v.end());
    }

    double max_ms(const std::string& stage) const {
        const auto& v = samples_.at(stage);
        return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end());
    }

    size_t num_samples(const std::string& stage) const {
        auto it = samples_.find(stage);
        return it == samples_.end() ? 0 : it->second.size();
    }

    /// All stage names currently recorded, in the order first inserted.
    const std::vector<std::string>& stage_order() const { return stage_order_; }

    /// Prints a table of every recorded stage to `os` (defaults to stdout).
    /// Call once for std::cout and again with an ofstream to also write the
    /// same table to a file.
    void print_summary(std::ostream& os = std::cout) const {
        os << std::left << std::setw(32) << "stage" << std::setw(8) << "n" << std::setw(12) << "mean(ms)"
           << std::setw(12) << "median(ms)" << std::setw(12) << "stddev(ms)" << std::setw(12) << "min(ms)"
           << std::setw(12) << "max(ms)" << "\n";
        for (const auto& stage : stage_order_) {
            os << std::left << std::setw(32) << stage << std::setw(8) << num_samples(stage) << std::setw(12)
               << mean_ms(stage) << std::setw(12) << median_ms(stage) << std::setw(12) << stddev_ms(stage)
               << std::setw(12) << min_ms(stage) << std::setw(12) << max_ms(stage) << "\n";
        }
    }

    /// Drops all currently recorded samples. Used to discard a warm-up phase
    /// before the "real" measured runs begin.
    void clear() {
        samples_.clear();
        stage_order_.clear();
    }

private:
    std::unordered_map<std::string, std::vector<double>> samples_;
    std::vector<std::string> stage_order_; // preserves insertion order for readable printing

    friend class ScopedTimer;
    void note_stage_seen(const std::string& stage) {
        if (samples_.find(stage) == samples_.end() || samples_.at(stage).empty()) {
            if (std::find(stage_order_.begin(), stage_order_.end(), stage) == stage_order_.end()) {
                stage_order_.push_back(stage);
            }
        }
    }
};

/// RAII helper: measures wall-clock time between construction and
/// destruction (or an explicit call to stop()) and records it into a
/// LatencyRecorder.
class ScopedTimer {
public:
    ScopedTimer(LatencyRecorder& recorder, std::string stage)
        : recorder_(recorder), stage_(std::move(stage)), start_(std::chrono::steady_clock::now()) {
        recorder_.note_stage_seen(stage_);
    }

    ~ScopedTimer() { stop(); }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /// Stops early and records; destructor becomes a no-op after this.
    void stop() {
        if (stopped_) return;
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        recorder_.add_sample(stage_, ms);
        stopped_ = true;
    }

private:
    LatencyRecorder& recorder_;
    std::string stage_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};

} // namespace psearch