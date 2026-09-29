/// Visualizer server over real sockets: the page, a 404, a missing file, and snapshots arriving as Server-Sent Events.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "test_framework.h"
#include "viz/server.h"

namespace {

// A folder holding just an index.html, so the test does not depend on the real page.
std::string make_root() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "mini_tpu_viz_root";
    std::filesystem::create_directories(root);
    std::ofstream(root / "index.html") << "hello page";
    std::filesystem::remove(root / "app.js");
    return root.string();
}

int connect_to(u32 port) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    timeval two_seconds{};
    two_seconds.tv_sec = 2;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &two_seconds, sizeof two_seconds);

    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_port        = htons(static_cast<std::uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

// Sends one GET and reads until `until` has arrived, the server closes, or 2 seconds pass.
std::string get(int fd, const std::string& path, const std::string& until) {
    const std::string request = "GET " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
    ::send(fd, request.data(), request.size(), 0);
    std::string reply;
    char buffer[4096];
    while (reply.find(until) == std::string::npos) {
        const ssize_t n = ::recv(fd, buffer, sizeof buffer, 0);
        if (n <= 0) {
            break;
        }
        reply.append(buffer, static_cast<std::size_t>(n));
    }
    return reply;
}

std::string read_until(int fd, const std::string& until) {
    std::string reply;
    char buffer[4096];
    while (reply.find(until) == std::string::npos) {
        const ssize_t n = ::recv(fd, buffer, sizeof buffer, 0);
        if (n <= 0) {
            break;
        }
        reply.append(buffer, static_cast<std::size_t>(n));
    }
    return reply;
}

bool has(const std::string& text, const std::string& piece) {
    return text.find(piece) != std::string::npos;
}

}  // namespace

TEST(viz_server_serves_the_page_and_404s) {
    VizServer server(make_root(), 0);   // 0: let the system pick a free port
    CHECK(server.port() != 0);

    const int page = connect_to(server.port());
    CHECK(page >= 0);
    const std::string reply = get(page, "/", "hello page");
    CHECK(has(reply, "200 OK"));
    CHECK(has(reply, "text/html"));
    CHECK(has(reply, "hello page"));
    ::close(page);

    const int missing = connect_to(server.port());
    CHECK(has(get(missing, "/../secret", "not found"), "404 Not Found"));   // only three fixed names are served
    ::close(missing);

    const int script = connect_to(server.port());
    CHECK(has(get(script, "/app.js", "missing"), "viz/app.js is missing"));
    ::close(script);
}

TEST(viz_server_streams_every_snapshot) {
    VizServer server(make_root(), 0);
    server.publish("{\"cycle\":1}");

    const int events = connect_to(server.port());
    const std::string first = get(events, "/events", "\"cycle\":1}\n\n");
    CHECK(has(first, "text/event-stream"));
    CHECK(has(first, "data: {\"cycle\":1}\n\n"));   // a page that connects late still gets the latest snapshot

    server.publish("{\"cycle\":2}");
    CHECK(has(read_until(events, "\"cycle\":2}\n\n"), "data: {\"cycle\":2}\n\n"));
    ::close(events);
}
