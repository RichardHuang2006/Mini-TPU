/// POSIX sockets, one thread per connection; only three fixed file names are served, so no path can escape viz/.

#include "viz/server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;   // Linux: a closed page must not raise SIGPIPE
#else
constexpr int kSendFlags = 0;              // macOS: SO_NOSIGPIPE is set on each socket instead
#endif

bool send_all(int fd, const std::string& data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, kSendFlags);
        if (n <= 0) {
            return false;
        }
        sent = sent + static_cast<std::size_t>(n);
    }
    return true;
}

// A client that stops reading or writing is dropped after 2 seconds instead of holding its thread forever.
void set_timeouts(int fd) {
    timeval two_seconds{};
    two_seconds.tv_sec = 2;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &two_seconds, sizeof two_seconds);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &two_seconds, sizeof two_seconds);
#ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof yes);
#endif
}

// The method and target (path and ?query) of the request's first line; false if the request never arrives whole.
bool read_request(int fd, std::string& method, std::string& target) {
    std::string head;
    char buffer[1024];
    while (head.find("\r\n\r\n") == std::string::npos && head.size() < 8192) {
        const ssize_t n = ::recv(fd, buffer, sizeof buffer, 0);
        if (n <= 0) {
            return false;
        }
        head.append(buffer, static_cast<std::size_t>(n));
    }

    const std::size_t method_end = head.find(' ');
    if (method_end == std::string::npos) {
        return false;
    }
    const std::size_t target_end = head.find(' ', method_end + 1);
    if (target_end == std::string::npos) {
        return false;
    }
    method = head.substr(0, method_end);
    target = head.substr(method_end + 1, target_end - method_end - 1);
    return true;
}

// The text after `key=` in a query such as "first=16&count=4", or "" if the key is missing.
std::string query_text(const std::string& query, const std::string& key) {
    const std::string wanted = key + "=";
    std::size_t at = 0;
    while (at < query.size()) {
        std::size_t end = query.find('&', at);
        if (end == std::string::npos) {
            end = query.size();
        }
        const std::string pair = query.substr(at, end - at);
        if (pair.compare(0, wanted.size(), wanted) == 0) {
            return pair.substr(wanted.size());
        }
        at = end + 1;
    }
    return "";
}

// The whole number after `key=`; false if it is missing, not all digits, or too long to be a row or cycle.
bool query_number(const std::string& query, const std::string& key, u64& value) {
    const std::string digits = query_text(query, key);
    const bool all_digits = !digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos;
    if (!all_digits || digits.size() > 18) {
        return false;
    }
    value = std::stoull(digits);
    return true;
}

bool read_file(const std::string& path, std::string& contents) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream text;
    text << file.rdbuf();
    contents = text.str();
    return true;
}

void send_bad_request(int fd, const std::string& why) {
    const std::string response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: " +
                                 std::to_string(why.size()) + "\r\nConnection: close\r\n\r\n" + why;
    send_all(fd, response);
}

// A button's reply: its text, sent as 400 when it starts with "error: ".
void send_reply(int fd, const std::string& reply) {
    if (reply.compare(0, 7, "error: ") == 0) {
        send_bad_request(fd, reply);
        return;
    }
    const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: " +
                             std::to_string(reply.size()) + "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    send_all(fd, head + reply);
}

void send_not_found(int fd, const std::string& why) {
    const std::string response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: " +
                                 std::to_string(why.size()) + "\r\nConnection: close\r\n\r\n" + why;
    send_all(fd, response);
}

}  // namespace

VizServer::VizServer(std::string root, u32 first_port) : root_(std::move(root)) {
    std::signal(SIGPIPE, SIG_IGN);   // belt and braces: a closed browser tab must never stop the simulator

    for (u32 port = first_port; port < first_port + 10; ++port) {
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) {
            throw std::runtime_error("visualizer: cannot create a socket");
        }
        int yes = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);

        sockaddr_in address{};
        address.sin_family      = AF_INET;
        address.sin_port        = htons(static_cast<std::uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // this machine only
        const bool bound = ::bind(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) == 0;
        if (bound && ::listen(fd, 16) == 0) {
            listen_fd_ = fd;
            break;
        }
        ::close(fd);
    }
    if (listen_fd_ < 0) {
        throw std::runtime_error("visualizer: ports " + std::to_string(first_port) + "-" + std::to_string(first_port + 9) +
                                 " are all taken");
    }

    // Port 0 asks the system for a free port; read back which one it chose.
    sockaddr_in bound_address{};
    socklen_t   length = sizeof bound_address;
    getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&bound_address), &length);
    port_ = ntohs(bound_address.sin_port);

    accept_thread_ = std::thread([this]() { accept_loop(); });
}

VizServer::~VizServer() {
    stopping_ = true;
    changed_.notify_all();
    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    ::close(listen_fd_);

    std::vector<std::thread> clients;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        clients.swap(client_threads_);
    }
    for (std::thread& client : clients) {
        client.join();
    }
}

u32 VizServer::port() const {
    return port_;
}

void VizServer::publish(const std::string& json, std::shared_ptr<const MachineState> state) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        latest_  = json;
        state_   = std::move(state);
        version_ = version_ + 1;
    }
    changed_.notify_all();
}

void VizServer::on_move(MoveHandler handler) {
    const std::lock_guard<std::mutex> lock(mutex_);
    move_ = std::move(handler);
}

void VizServer::on_jump(JumpHandler handler) {
    const std::lock_guard<std::mutex> lock(mutex_);
    jump_ = std::move(handler);
}

// Waits for connections, checking every 100 ms whether the server is shutting down.
void VizServer::accept_loop() {
    while (!stopping_) {
        pollfd waiting{};
        waiting.fd     = listen_fd_;
        waiting.events = POLLIN;
        if (::poll(&waiting, 1, 100) <= 0) {
            continue;
        }

        const int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0) {
            continue;
        }
        set_timeouts(client);

        const std::lock_guard<std::mutex> lock(mutex_);
        client_threads_.emplace_back([this, client]() { serve(client); });
    }
}

void VizServer::serve(int fd) {
    std::string method;
    std::string target;
    if (!read_request(fd, method, target)) {
        ::close(fd);
        return;
    }
    std::string path  = target;
    std::string query;
    const std::size_t mark = target.find('?');
    if (mark != std::string::npos) {
        path  = target.substr(0, mark);
        query = target.substr(mark + 1);
    }

    // The only POSTs are /move and /jump, so a page can step the machine but never load a file; everything else is a GET.
    if (method == "POST" && path == "/move") {
        serve_move(fd, query);
    } else if (method == "POST" && path == "/jump") {
        serve_jump(fd, query);
    } else if (method != "GET") {
        send_not_found(fd, "not found");
    } else if (path == "/rows" || path == "/pes") {
        serve_data(fd, path, query);
    } else if (path == "/events") {
        serve_events(fd);
    } else if (path == "/" || path == "/index.html") {
        serve_file(fd, "index.html", "text/html; charset=utf-8");
    } else if (path == "/style.css") {
        serve_file(fd, "style.css", "text/css; charset=utf-8");
    } else if (path == "/app.js") {
        serve_file(fd, "app.js", "text/javascript; charset=utf-8");
    } else {
        send_not_found(fd, "not found");
    }
    ::close(fd);
}

void VizServer::serve_file(int fd, const std::string& name, const std::string& type) {
    std::string body;
    if (!read_file(root_ + "/" + name, body)) {
        send_not_found(fd, "viz/" + name + " is missing");
        return;
    }
    const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: " + type + "\r\nContent-Length: " + std::to_string(body.size()) +
                             "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    send_all(fd, head + body);
}

// Exact values from the latest state copy, as raw little-endian bytes; X-Cycle says which cycle they belong to.
void VizServer::serve_data(int fd, const std::string& path, const std::string& query) {
    std::shared_ptr<const MachineState> state;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        state = state_;
    }
    if (!state) {
        send_bad_request(fd, "no machine state yet");
        return;
    }

    u64 first = 0;
    u64 count = 0;
    if (!query_number(query, "first", first) || !query_number(query, "count", count)) {
        send_bad_request(fd, "expected ?first=N&count=N");
        return;
    }

    std::string body;
    bool ok = false;
    if (path == "/rows") {
        ok = rows_bytes(*state, query_text(query, "memory"), first, count, body);
    } else {
        ok = pes_bytes(*state, first, count, body);
    }
    if (!ok) {
        send_bad_request(fd, "no such memory, or the range runs past it");
        return;
    }

    const std::string head = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: " +
                             std::to_string(body.size()) + "\r\nX-Cycle: " + std::to_string(state->cycle) +
                             "\r\nAccess-Control-Expose-Headers: X-Cycle\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
    send_all(fd, head + body);
}

// A button press: the handler runs outside the lock, so publishing the snapshot it causes cannot deadlock.
void VizServer::serve_move(int fd, const std::string& query) {
    MoveHandler move;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        move = move_;
    }
    if (!move) {
        send_bad_request(fd, "no machine to move");
        return;
    }

    send_reply(fd, move(query_text(query, "direction"), query_text(query, "unit")));
}

// A jump to a cycle, run outside the lock like a button press.
void VizServer::serve_jump(int fd, const std::string& query) {
    JumpHandler jump;
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        jump = jump_;
    }
    u64 cycle = 0;
    if (!jump || !query_number(query, "cycle", cycle)) {
        send_bad_request(fd, "expected ?cycle=N");
        return;
    }
    send_reply(fd, jump(cycle));
}

// Every new snapshot goes out as `data: <json>`; a comment line every second finds pages that have gone away.
void VizServer::serve_events(int fd) {
    const std::string head =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n";
    if (!send_all(fd, head)) {
        return;
    }

    u64 seen = 0;
    while (!stopping_) {
        std::string message = ": still here\n\n";
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait_for(lock, std::chrono::seconds(1), [this, seen]() { return stopping_ || version_ != seen; });
            if (stopping_) {
                return;
            }
            if (version_ != seen) {
                message = "data: " + latest_ + "\n\n";
                seen = version_;
            }
        }
        if (!send_all(fd, message)) {
            return;
        }
    }
}
