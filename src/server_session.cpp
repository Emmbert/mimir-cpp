#include "server_session.hpp"

namespace psearch {

    size_t SessionStore::SessionIdHash::operator()(const SessionId& id) const {
        // Simple FNV-1a over the 16 bytes. Session IDs are already
        // cryptographically random (generate_fresh_seed), so this hash itself
        // doesn't need to be collision-resistant as a security property -- it
        // only needs to distribute reasonably well for the map.
        size_t h = 14695981039346656037ULL;
        for (uint8_t b : id) {
            h ^= b;
            h *= 1099511628211ULL;
        }
        return h;
    }

    SessionStore::SessionStore(std::chrono::steady_clock::duration ttl) : ttl_(ttl) {}

    void SessionStore::register_session(const SessionId& id, ClientPublicMaterial pub) {
        std::lock_guard<std::mutex> lock(mutex_);
        sessions_[id] = Session{std::move(pub), std::chrono::steady_clock::now()};
    }

    std::optional<ClientPublicMaterial> SessionStore::get_and_touch(const SessionId& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = sessions_.find(id);
        if (it == sessions_.end()) {
            return std::nullopt;
        }
        auto now = std::chrono::steady_clock::now();
        if (now - it->second.last_used > ttl_) {
            sessions_.erase(it);
            return std::nullopt;
        }
        it->second.last_used = now;
        return it->second.pub;
    }

} // namespace psearch