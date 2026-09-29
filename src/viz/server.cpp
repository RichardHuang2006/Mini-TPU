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

// The path of a GET request, without any ?query; "" for anything else.
std::string request_path(int fd) {
    std::string head;
    char buffer[1024];
    while (head.find("\r\n\r\n") == std::string::npos && head.size() < 8192) {
        const ssize_t n = ::recv(fd, buffer, sizeof buffer, 0);
        if (n <= 0) {
            return "";
        }
        head.append(buffer, static_cast<std::size_t>(n));
    }

    if (head.compare(0, 4, "GET ") != 0) {
        return "";
    }
    const std::size_t end = head.find(' ', 4);
    if (end == std::string::npos) {
        return "";
    }
    std::string path = head.substr(4, end - 4);
    const std::size_t query = path.find('?');
    if (query != std::string::npos) {
        path = path.substr(0, query);
    }
    return path;
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

void VizServer::publish(const std::string& json) {
    {
        const std::lock_guard<std::mutex> lock(mutex_);
        latest_  = json;
        version_ = version_ + 1;
    }
    changed_.notify_all();
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
    const std::string path = request_path(fd);
    if (path == "/events") {
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
