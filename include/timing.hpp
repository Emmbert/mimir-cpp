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
///
/// Nesting is tracked automatically: a stage is shown as a child of every
/// ScopedTimer that is still open when the stage is first seen (either via
/// its own ScopedTimer or via add_sample). Children are printed indented with
/// "-> ", so stage names must NOT carry a manual "  -> " prefix:
///
///   server_processing
///   -> query unpacking
///     -> query stream unpacking
///   -> scoring calculations
///     -> score computation      <- rec.add_sample(...) inside t_scoring's scope
///
/// Not thread-safe: create timers and call add_sample from one thread only
/// (all benchmarks do this outside their OpenMP regions).
class LatencyRecorder {
public:
    void add_sample(const std::string& stage, double milliseconds) {
        note_stage_seen(stage);
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
    ///
    /// The stage column is as wide as the longest (indented) label, and all
    /// numbers are right-aligned with a fixed 3 decimals (i.e. microsecond
    /// resolution), so columns always line up.
    void print_summary(std::ostream& os = std::cout) const {
        std::vector<std::string> labels;
        labels.reserve(stage_order_.size());
        size_t label_width = 5; // "stage"
        for (const auto& stage : stage_order_) {
            labels.push_back(indented_label(stage));
            label_width = std::max(label_width, labels.back().size());
        }
        label_width += 2;

        constexpr int kCountWidth = 6;
        constexpr int kNumWidth = 13;

        std::ios_base::fmtflags saved_flags = os.flags();
        std::streamsize saved_precision = os.precision();

        os << std::left << std::setw(static_cast<int>(label_width)) << "stage" << std::right
           << std::setw(kCountWidth) << "n" << std::setw(kNumWidth) << "mean(ms)" << std::setw(kNumWidth)
           << "median(ms)" << std::setw(kNumWidth) << "stddev(ms)" << std::setw(kNumWidth) << "min(ms)"
           << std::setw(kNumWidth) << "max(ms)" << "\n";

        os << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < stage_order_.size(); ++i) {
            const auto& stage = stage_order_[i];
            os << std::left << std::setw(static_cast<int>(label_width)) << labels[i] << std::right
               << std::setw(kCountWidth) << num_samples(stage) << std::setw(kNumWidth) << mean_ms(stage)
               << std::setw(kNumWidth) << median_ms(stage) << std::setw(kNumWidth) << stddev_ms(stage)
               << std::setw(kNumWidth) << min_ms(stage) << std::setw(kNumWidth) << max_ms(stage) << "\n";
        }

        os.flags(saved_flags);
        os.precision(saved_precision);
    }

    /// Drops all currently recorded samples. Used to discard a warm-up phase
    /// before the "real" measured runs begin.
    void clear() {
        samples_.clear();
        stage_order_.clear();
        stage_depth_.clear();
    }

private:
    std::unordered_map<std::string, std::vector<double>> samples_;
    std::vector<std::string> stage_order_;             // preserves insertion order for readable printing
    std::unordered_map<std::string, int> stage_depth_; // nesting depth when the stage was first seen
    int open_timers_ = 0;                              // ScopedTimers currently running

    friend class ScopedTimer;

    /// Registers `stage` (once) at the current nesting depth.
    void note_stage_seen(const std::string& stage) {
        if (stage_depth_.emplace(stage, open_timers_).second) {
            stage_order_.push_back(stage);
        }
    }

    std::string indented_label(const std::string& stage) const {
        int depth = stage_depth_.at(stage);
        if (depth == 0) {
            return stage;
        }
        return std::string(static_cast<size_t>(2 * (depth - 1)), ' ') + "-> " + stage;
    }
};

/// RAII helper: measures wall-clock time between construction and
/// destruction (or an explicit call to stop()) and records it into a
/// LatencyRecorder.
class ScopedTimer {
public:
    ScopedTimer(LatencyRecorder& recorder, std::string stage)
        : recorder_(recorder), stage_(std::move(stage)) {
        recorder_.note_stage_seen(stage_); // registered at the depth of its parent timers
        ++recorder_.open_timers_;
        start_ = std::chrono::steady_clock::now();
    }

    ~ScopedTimer() { stop(); }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

    /// Stops early and records; destructor becomes a no-op after this.
    void stop() {
        if (stopped_) return;
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        --recorder_.open_timers_;
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