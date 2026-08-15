#include "network.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace psearch {

Socket::~Socket() {
    if (fd_ >= 0) {
        ::close(fd_);
    }
}

Socket::Socket(Socket&& other) noexcept : fd_(other.fd_) {
    other.fd_ = -1;
}

Socket& Socket::operator=(Socket&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) {
            ::close(fd_);
        }
        fd_ = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

void Socket::send_all(const uint8_t* data, size_t size) const {
    size_t sent = 0;
    while (sent < size) {
        ssize_t n = ::send(fd_, data + sent, size - sent, 0);
        if (n < 0) {
            throw std::runtime_error(std::string("Socket::send_all: send() failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error("Socket::send_all: connection closed mid-send");
        }
        sent += static_cast<size_t>(n);
    }
}

void Socket::recv_all(uint8_t* out, size_t size) const {
    size_t received = 0;
    while (received < size) {
        ssize_t n = ::recv(fd_, out + received, size - received, 0);
        if (n < 0) {
            throw std::runtime_error(std::string("Socket::recv_all: recv() failed: ") + std::strerror(errno));
        }
        if (n == 0) {
            throw std::runtime_error("Socket::recv_all: connection closed before expected data arrived");
        }
        received += static_cast<size_t>(n);
    }
}

void Socket::send_message(const Message& msg) const {
    uint8_t type_byte = static_cast<uint8_t>(msg.type);
    uint32_t len = static_cast<uint32_t>(msg.payload.size());

    send_all(&type_byte, 1);
    send_all(reinterpret_cast<const uint8_t*>(&len), sizeof(len));
    if (!msg.payload.empty()) {
        send_all(msg.payload.data(), msg.payload.size());
    }
}

Message Socket::recv_message() const {
    uint8_t type_byte;
    recv_all(&type_byte, 1);

    uint32_t len;
    recv_all(reinterpret_cast<uint8_t*>(&len), sizeof(len));

    Message msg;
    msg.type = static_cast<MessageType>(type_byte);
    msg.payload.resize(len);
    if (len > 0) {
        recv_all(msg.payload.data(), len);
    }
    return msg;
}

Socket connect_to_server(const std::string& host, uint16_t port) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* result = nullptr;
    std::string port_str = std::to_string(port);
    int rc = ::getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result);
    if (rc != 0) {
        throw std::runtime_error("connect_to_server: getaddrinfo failed for " + host + ":" + port_str + ": " +
                                  ::gai_strerror(rc));
    }

    int fd = -1;
    for (addrinfo* p = result; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) {
            continue;
        }
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(result);

    if (fd < 0) {
        throw std::runtime_error("connect_to_server: could not connect to " + host + ":" + port_str);
    }
    return Socket(fd);
}

Socket create_listening_socket(uint16_t port, int backlog) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("create_listening_socket: socket() failed: ") + std::strerror(errno));
    }

    int reuse = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        throw std::runtime_error(std::string("create_listening_socket: bind() failed on port ") +
                                  std::to_string(port) + ": " + std::strerror(errno));
    }

    if (::listen(fd, backlog) < 0) {
        ::close(fd);
        throw std::runtime_error(std::string("create_listening_socket: listen() failed: ") + std::strerror(errno));
    }

    return Socket(fd);
}

Socket accept_connection(const Socket& listening_socket) {
    sockaddr_in client_addr{};
    socklen_t addr_len = sizeof(client_addr);
    int fd = ::accept(listening_socket.fd(), reinterpret_cast<sockaddr*>(&client_addr), &addr_len);
    if (fd < 0) {
        throw std::runtime_error(std::string("accept_connection: accept() failed: ") + std::strerror(errno));
    }
    return Socket(fd);
}

} // namespace psearch
