#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace psearch {

/// Message types for the length-prefixed wire protocol: [1-byte type]
/// [4-byte payload length][payload]. See wire_protocol.hpp for what each
/// payload actually contains.
enum class MessageType : uint8_t {
    Registration = 1,     // client -> server: session_id + SeededClientPublicMaterial
    Query = 2,            // client -> server: session_id + SeededQuery
    QueryResponse = 3,    // server -> client: QueryResponse
    Error = 4,            // server -> client: a UTF-8 error string
    RegistrationAck = 5,  // server -> client: empty payload, registration succeeded
};

struct Message {
    MessageType type;
    std::vector<uint8_t> payload;
};

/// Thin RAII wrapper around a POSIX socket file descriptor. Closes on
/// destruction; move-only (a socket fd shouldn't have two owners).
class Socket {
public:
    Socket() = default;
    explicit Socket(int fd) : fd_(fd) {}
    ~Socket();

    Socket(const Socket&) = delete;
    Socket& operator=(const Socket&) = delete;
    Socket(Socket&& other) noexcept;
    Socket& operator=(Socket&& other) noexcept;

    int fd() const { return fd_; }
    bool valid() const { return fd_ >= 0; }

    /// Sends exactly `data.size()` bytes, retrying on partial writes.
    /// Throws std::runtime_error on any socket error.
    void send_all(const uint8_t* data, size_t size) const;

    /// Reads exactly `size` bytes into `out` (must already be sized),
    /// retrying on partial reads. Throws std::runtime_error on error or if
    /// the connection closes before `size` bytes arrive.
    void recv_all(uint8_t* out, size_t size) const;

    /// Sends one framed message: [1-byte type][4-byte length][payload].
    void send_message(const Message& msg) const;

    /// Blocks until one full framed message has arrived.
    Message recv_message() const;

private:
    int fd_ = -1;
};

/// Client-side: opens a TCP connection to host:port. Throws
/// std::runtime_error on failure (DNS resolution, connect refused, etc).
Socket connect_to_server(const std::string& host, uint16_t port);

/// Server-side: binds and listens on port (all interfaces). Throws
/// std::runtime_error on failure (port already in use, permission denied
/// for privileged ports, etc).
Socket create_listening_socket(uint16_t port, int backlog = 16);

/// Server-side: blocks until one client connects, returns the accepted
/// connection. Call in a loop; hand each accepted Socket to its own thread
/// for a multi-client server.
Socket accept_connection(const Socket& listening_socket);

} // namespace psearch