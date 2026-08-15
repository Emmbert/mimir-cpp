#pragma once
#include <chrono>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "key_material.hpp"
#include "wire_protocol.hpp" // SessionId

namespace psearch {

    /// Thread-safe in-memory session store: session_id -> reconstructed
    /// ClientPublicMaterial, evicted a fixed time after last use (10 minutes by
    /// default). Lazy eviction only (checked on lookup/insert) -- no background
    /// reaper thread. Simplest correct option for a first version; a real
    /// long-running deployment might eventually want periodic sweeping too, to
    /// reclaim memory from sessions that register but never come back to
    /// query -- that's a deliberate future addition, not built by default here.
    class SessionStore {
    public:
        explicit SessionStore(std::chrono::steady_clock::duration ttl = std::chrono::minutes(10));

        /// Stores/overwrites a session's public material and resets its TTL.
        void register_session(const SessionId& id, ClientPublicMaterial pub);

        /// Returns the session's ClientPublicMaterial if it exists and hasn't
        /// expired (refreshing its last-used time on success), or std::nullopt
        /// otherwise -- callers should treat nullopt as "ask the client to
        /// register again".
        std::optional<ClientPublicMaterial> get_and_touch(const SessionId& id);

    private:
        struct Session {
            ClientPublicMaterial pub;
            std::chrono::steady_clock::time_point last_used;
        };

        struct SessionIdHash {
            size_t operator()(const SessionId& id) const;
        };

        std::mutex mutex_;
        std::chrono::steady_clock::duration ttl_;
        std::unordered_map<SessionId, Session, SessionIdHash> sessions_;
    };

} // namespace psearch