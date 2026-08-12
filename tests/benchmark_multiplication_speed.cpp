// benchmark_multiplication_speed.cpp
//
// Isolates the ONE thing test_rlwe_plaintext_multiplication.cpp and
// test_rlwe_plaintext_multiplication_ntt.cpp differ on — coefficient-form vs
// eval-form multiplication — from everything else (keygen, LWE encryption,
// key switching), which both of those tests redo every iteration and which
// costs far more than a single degree-n multiplication. That shared,
// dominant cost is why running both tests 200x showed no visible speed
// difference: you were measuring ~200x keygen, with the multiplication
// method being a rounding error on top of it either way.
//
// Here, keys and the query ciphertext are built ONCE, outside the timed
// region. Database entries for the eval-form path are also pre-converted
// before timing starts — exactly like a real stored database would already
// be in eval form by the time a query arrives, per db_polynomial.hpp's
// build_random_database_polynomial_eval_form. Only the actual multiply is
// timed.
//
// Run directly to see the printed timings (ctest swallows stdout on a
// passing test):
//   ./benchmark_multiplication_speed

#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "db_polynomial.hpp"
#include "fhe_deck.h"
#include "key_material.hpp"
#include "params.hpp"

using namespace FHEDeck;
using namespace psearch;

namespace {

double milliseconds_between(std::chrono::steady_clock::time_point start,
                             std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

TEST(MultiplicationSpeedBenchmark, CoefficientFormVsEvalForm) {
    Params params = Params::make_test_params();
    ASSERT_FALSE(products_can_overflow(params));
    CryptoContext ctx = CryptoContext::from_params(params);

    // --- Setup: keygen, one query ciphertext, N database entries in BOTH
    // forms. None of this is timed — it happens once, like a real deployment
    // would do keygen once per client session and database setup once ever.
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    ClientPublicMaterial pub = generate_client_public_material(ctx, secret);

    std::mt19937_64 rng(std::random_device{}());
    SignedValue message = sample_signed_value(params, rng);
    LWECT lwe_ct = secret.lwe_sk->encode_and_encrypt(message.reduced, ctx.encoding);

    RLWECT rlwe_ct(ctx.rlwe_param);
    pub.lwe_to_rlwe_ksk->lwe_to_rlwe_key_switch(rlwe_ct, lwe_ct);
    RLWECTEvalForm rlwe_ct_eval(rlwe_ct); // ciphertext eval-form, built once

    constexpr int kNumDatabaseEntries = 500; // number of multiplications timed per path
    constexpr int kWarmupEntries = 20;       // discarded, same reasoning as any benchmark warm-up

    std::vector<Polynomial> db_coef_form;
    std::vector<DatabasePolynomialEvalForm> db_eval_form;
    db_coef_form.reserve(kNumDatabaseEntries + kWarmupEntries);
    db_eval_form.reserve(kNumDatabaseEntries + kWarmupEntries);

    for (int i = 0; i < kNumDatabaseEntries + kWarmupEntries; ++i) {
        DatabasePolynomial coef_db = build_random_database_polynomial(params, rng);
        db_coef_form.push_back(std::move(coef_db.poly));

        db_eval_form.push_back(build_random_database_polynomial_eval_form(ctx, params, rng));
    }

    // --- Warm-up (discarded): first calls into a fresh mul_engine/NTT path
    // can carry one-time setup cost (e.g. table generation) that has nothing
    // to do with steady-state per-multiplication cost.
    for (int i = 0; i < kWarmupEntries; ++i) {
        RLWECT out(ctx.rlwe_param);
        rlwe_ct.mul(out, db_coef_form[static_cast<size_t>(i)]);

        RLWECTEvalForm out_eval(ctx.rlwe_param);
        rlwe_ct_eval.mul(out_eval, *db_eval_form[static_cast<size_t>(i)].poly_eval);
    }

    // --- Timed: coefficient-form path ---------------------------------------
    auto coef_start = std::chrono::steady_clock::now();
    for (int i = kWarmupEntries; i < kWarmupEntries + kNumDatabaseEntries; ++i) {
        RLWECT out(ctx.rlwe_param);
        rlwe_ct.mul(out, db_coef_form[static_cast<size_t>(i)]);
    }
    auto coef_end = std::chrono::steady_clock::now();
    double coef_total_ms = milliseconds_between(coef_start, coef_end);

    // --- Timed: eval-form path -----------------------------------------------
    auto eval_start = std::chrono::steady_clock::now();
    for (int i = kWarmupEntries; i < kWarmupEntries + kNumDatabaseEntries; ++i) {
        RLWECTEvalForm out_eval(ctx.rlwe_param);
        rlwe_ct_eval.mul(out_eval, *db_eval_form[static_cast<size_t>(i)].poly_eval);
    }
    auto eval_end = std::chrono::steady_clock::now();
    double eval_total_ms = milliseconds_between(eval_start, eval_end);

    std::cout << "=== Multiplication speed (n=" << params.n << ", " << kNumDatabaseEntries
              << " multiplications, " << kWarmupEntries << " warm-up discarded) ===\n";
    std::cout << "coefficient-form: " << coef_total_ms << " ms total, "
              << (coef_total_ms / kNumDatabaseEntries) << " ms/op\n";
    std::cout << "eval-form:        " << eval_total_ms << " ms total, "
              << (eval_total_ms / kNumDatabaseEntries) << " ms/op\n";
    std::cout << "speedup: " << (coef_total_ms / eval_total_ms) << "x\n";
}
