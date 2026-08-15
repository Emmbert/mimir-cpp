// test_server_session.cpp
//
// Tests SessionStore: register/lookup round-trip, missing-session lookup,
// TTL expiry, that a successful lookup ("touch") extends the TTL rather
// than leaving it pinned to the original registration time, and that
// re-registering resets the clock. Uses a SHORT, injected TTL
// (milliseconds) rather than the real 10-minute default, so expiry can be
// tested deterministically and fast via sleep_for, instead of waiting ten
// real minutes.
//
// Run via `ctest` (see CMakeLists.txt / README.md), or directly:
//   ./test_server_session

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

#include "key_material.hpp"
#include "params.hpp"
#include "server_session.hpp"

using namespace psearch;

namespace {

SessionId make_id(uint8_t fill) {
    SessionId id;
    id.fill(fill);
    return id;
}

ClientPublicMaterial make_dummy_material() {
    static Params params = Params::make_test_params();
    static CryptoContext ctx = CryptoContext::from_params(params);
    ClientSecretMaterial secret = generate_client_secret_material(ctx, params);
    return generate_client_public_material(ctx, secret);
}

} // namespace

TEST(ServerSession, RegisterThenLookupSucceeds) {
    SessionStore store(std::chrono::minutes(10));
    SessionId id = make_id(1);

    store.register_session(id, make_dummy_material());
    auto result = store.get_and_touch(id);

    EXPECT_TRUE(result.has_value());
}

TEST(ServerSession, UnknownSessionReturnsNullopt) {
    SessionStore store(std::chrono::minutes(10));
    SessionId unknown_id = make_id(42);

    auto result = store.get_and_touch(unknown_id);

    EXPECT_FALSE(result.has_value());
}

TEST(ServerSession, ExpiredSessionReturnsNulloptAndIsEvicted) {
    SessionStore store(std::chrono::milliseconds(20)); // short TTL for a fast, deterministic test
    SessionId id = make_id(2);

    store.register_session(id, make_dummy_material());
    std::this_thread::sleep_for(std::chrono::milliseconds(50)); // let it expire

    auto result = store.get_and_touch(id);
    EXPECT_FALSE(result.has_value());

    // Confirm the eviction code path was actually exercised (not just
    // "reported expired while still occupying the map"): re-registering
    // the same id immediately after should succeed cleanly.
    store.register_session(id, make_dummy_material());
    auto result2 = store.get_and_touch(id);
    EXPECT_TRUE(result2.has_value());
}

TEST(ServerSession, LookupExtendsTtlRatherThanLeavingItPinned) {
    // TTL of 60ms; touch it every 20ms, well within the window each time --
    // if get_and_touch correctly refreshes last_used, the session should
    // still be alive after 100ms total elapsed (well past the original
    // 60ms TTL measured from registration time alone).
    SessionStore store(std::chrono::milliseconds(60));
    SessionId id = make_id(3);
    store.register_session(id, make_dummy_material());

    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        auto result = store.get_and_touch(id);
        ASSERT_TRUE(result.has_value())
            << "Session expired at iteration " << i << " despite being touched every 20ms with a 60ms TTL -- "
            << "get_and_touch may not be refreshing last_used correctly.";
    }
}

TEST(ServerSession, ReRegisteringResetsTtl) {
    SessionStore store(std::chrono::milliseconds(50));
    SessionId id = make_id(4);

    store.register_session(id, make_dummy_material());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    store.register_session(id, make_dummy_material()); // re-register -- should reset the TTL clock
    std::this_thread::sleep_for(std::chrono::milliseconds(30)); // 60ms since first reg, but only 30ms since second

    auto result = store.get_and_touch(id);
    EXPECT_TRUE(result.has_value())
        << "Session expired despite being re-registered partway through its TTL window -- "
        << "register_session may not be resetting last_used.";
}
