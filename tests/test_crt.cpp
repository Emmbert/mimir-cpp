// test_crt.cpp
//
// Tests crt_split/crt_recompose: the round trip split -> recompose must
// return the original value exactly, for every value in the combined
// modulus range. Also specifically checks the boundary values (0, and the
// largest representable value) since off-by-one errors at range edges are
// the most common way modular arithmetic silently breaks.
//
// Uses Params::make_test_params_component_rings() for the "real"
// comp_ring_modulus tests (the exhaustive small-modulus test below is pure
// algorithm verification and deliberately independent of any Params). That
// factory respects MIMIR_TEST_PARAMS_FILE the same way every other test's
// params factory does -- if set, and the loaded file has
// num_component_rings == 1 (not actually CRT), CRT-specific tests SKIP
// rather than fail, since they're not applicable to a non-CRT parameter set.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_crt
//   MIMIR_TEST_PARAMS_FILE=path/to/crt_params.json ./test_crt

#include <gtest/gtest.h>

#include <random>

#include "crt.hpp"
#include "params.hpp"

using namespace psearch;

namespace {

void check_round_trip(int64_t v, int64_t comp_ring_modulus) {
    auto [r1, r2] = crt_split(v, comp_ring_modulus);
    int64_t recomposed = crt_recompose(r1, r2, comp_ring_modulus);
    EXPECT_EQ(recomposed, v) << "v=" << v << " comp_ring_modulus=" << comp_ring_modulus << " r1=" << r1
                              << " r2=" << r2;
}

/// Shared fixture for every test that needs a real, project-relevant
/// comp_ring_modulus (as opposed to the exhaustive small-modulus test,
/// which doesn't need Params at all). Loads params once in SetUp() and
/// skips the whole test if the loaded parameters aren't actually CRT --
/// e.g. if MIMIR_TEST_PARAMS_FILE points at a non-CRT file.
class CrtWithProjectParams : public ::testing::Test {
protected:
    void SetUp() override {
        params_ = Params::make_test_params_component_rings();
        if (params_.num_component_rings != 2) {
            GTEST_SKIP() << "Loaded parameters have num_component_rings=" << params_.num_component_rings
                         << " (not CRT) -- this test only applies to two-component-ring parameters. "
                         << "If MIMIR_TEST_PARAMS_FILE is set, point it at a file with r_NUM_COMP_RINGS=2, "
                         << "or unset it to use the built-in CRT test defaults.";
        }
    }

    Params params_;
};

} // namespace

TEST(Crt, RoundTripExhaustiveSmallModulus) {
    // Small enough to check every single value in the combined range, not
    // just a sample -- comp_ring_modulus=11 -> p1=11, p2=10, combined=110.
    // Deliberately independent of Params/MIMIR_TEST_PARAMS_FILE: this is
    // pure algorithm verification, not tied to any real deployment scenario.
    int64_t comp_ring_modulus = 11;
    int64_t combined = crt_combined_modulus(comp_ring_modulus);
    ASSERT_EQ(combined, 110);
    for (int64_t v = 0; v < combined; ++v) {
        check_round_trip(v, comp_ring_modulus);
    }
}

TEST_F(CrtWithProjectParams, RoundTripRandomSample) {
    int64_t comp_ring_modulus = params_.comp_ring_modulus;
    int64_t combined = params_.combined_component_ring_modulus;

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, combined - 1);
    for (int i = 0; i < 200000; ++i) {
        check_round_trip(dist(rng), comp_ring_modulus);
    }
}

TEST_F(CrtWithProjectParams, BoundaryValues) {
    int64_t comp_ring_modulus = params_.comp_ring_modulus;
    int64_t combined = params_.combined_component_ring_modulus;

    check_round_trip(0, comp_ring_modulus);
    check_round_trip(1, comp_ring_modulus);
    check_round_trip(combined - 1, comp_ring_modulus); // the largest representable value
    check_round_trip(comp_ring_modulus - 1, comp_ring_modulus);     // r1 wraps to 0 here
    check_round_trip(comp_ring_modulus, comp_ring_modulus);         // r1=0, r2=1 boundary
    check_round_trip(comp_ring_modulus - 2, comp_ring_modulus);     // r2 wraps to 0 here (p2 = p1-1)
}

TEST_F(CrtWithProjectParams, SplitComponentsAreInExpectedRanges) {
    int64_t comp_ring_modulus = params_.comp_ring_modulus;
    int64_t combined = params_.combined_component_ring_modulus;

    std::mt19937_64 rng(7);
    std::uniform_int_distribution<int64_t> dist(0, combined - 1);
    for (int i = 0; i < 10000; ++i) {
        int64_t v = dist(rng);
        auto [r1, r2] = crt_split(v, comp_ring_modulus);
        EXPECT_GE(r1, 0);
        EXPECT_LT(r1, comp_ring_modulus);
        EXPECT_GE(r2, 0);
        EXPECT_LT(r2, comp_ring_modulus - 1);
    }
}

TEST_F(CrtWithProjectParams, DifferentValuesProduceDifferentComponents) {
    // Sanity check that split() is actually doing something meaningful --
    // two distinct values should (almost always) produce distinct
    // (r1, r2) pairs, since the whole point of CRT is a bijection between
    // [0, combined) and Z_p1 x Z_p2.
    int64_t comp_ring_modulus = params_.comp_ring_modulus;
    auto a = crt_split(12345 % crt_combined_modulus(comp_ring_modulus), comp_ring_modulus);
    auto b = crt_split(12346 % crt_combined_modulus(comp_ring_modulus), comp_ring_modulus);
    EXPECT_NE(a, b);
}

TEST_F(CrtWithProjectParams, CombinedModulusMeetsPlaintextModulusLowerBound) {
    // Sanity check on the params factory itself -- derive_dependent_parameters()
    // already validates this at construction time (it would have thrown
    // otherwise), but asserting it again here documents WHY this parameter
    // set is valid, directly alongside the CRT tests that depend on it.
    int64_t combined = params_.combined_component_ring_modulus;
    EXPECT_GE(combined, params_.plaintext_modulus);
}