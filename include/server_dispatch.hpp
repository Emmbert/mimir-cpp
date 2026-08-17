#pragma once
#include "params.hpp"
#include "server_db.hpp"
#include "server_session.hpp"
#include "network.hpp"

namespace psearch {

/// Pure dispatch: given one incoming Message, produces the response
/// Message. No socket I/O anywhere in here -- server.cpp's accept loop
/// calls this after reading a request and before writing the response;
/// tests can call it directly without any real network connection.
Message handle_message(const Message& request, SessionStore& sessions, const CryptoContext& ctx,
                        const Params& params, const ServerDatabase& db);

} // namespace psearch
