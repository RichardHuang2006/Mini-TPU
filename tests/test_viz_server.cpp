/// Visualizer server over real sockets: the page, a 404, snapshots as Server-Sent Events, exact values and button presses and jumps.

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
#include "isa/asm.h"
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

// Sends one request and reads until `until` has arrived, the server closes, or 2 seconds pass.
std::string send_request(int fd, const std::string& method, const std::string& path, const std::string& until) {
    const std::string request = method + " " + path + " HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 0\r\n\r\n";
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

std::string get(int fd, const std::string& path, const std::string& until) {
    return send_request(fd, "GET", path, until);
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
    Tpu tpu;
    server.publish("{\"cycle\":1}", capture(tpu));

    const int events = connect_to(server.port());
    const std::string first = get(events, "/events", "\"cycle\":1}\n\n");
    CHECK(has(first, "text/event-stream"));
    CHECK(has(first, "data: {\"cycle\":1}\n\n"));   // a page that connects late still gets the latest snapshot

    server.publish("{\"cycle\":2}", capture(tpu));
    CHECK(has(read_until(events, "\"cycle\":2}\n\n"), "data: {\"cycle\":2}\n\n"));
    ::close(events);
}

TEST(viz_server_sends_exact_values) {
    VizServer server(make_root(), 0);
    Tpu tpu;
    tpu.load(assemble(".host 0\n7 -8 9\nRead_Host_Memory host=0 ub=5 rows=1\nHalt\n"));
    tpu.run_to_halt();
    const auto state = capture(tpu);
    server.publish("{}", state);

    std::string expected;
    rows_bytes(*state, "ub", 5, 1, expected);
    const int rows = connect_to(server.port());
    const std::string reply = get(rows, "/rows?memory=ub&first=5&count=1", "\r\n\r\n");
    const std::string body  = reply.substr(reply.find("\r\n\r\n") + 4) + read_until(rows, "never arrives");
    ::close(rows);
    CHECK(has(reply, "200 OK"));
    CHECK(has(reply, "X-Cycle: " + std::to_string(tpu.cycle())));
    CHECK(body == expected);
    CHECK_EQ(static_cast<int>(static_cast<i8>(body[1])), -8);

    const int bad = connect_to(server.port());
    CHECK(has(get(bad, "/rows?memory=ub&first=98304&count=1", "past it"), "400 Bad Request"));
    ::close(bad);
}

TEST(viz_server_passes_button_presses_to_the_machine) {
    VizServer server(make_root(), 0);
    std::string pressed;
    server.on_move([&pressed](const std::string& direction, const std::string& unit) {
        pressed = direction + " " + unit;
        if (direction == "up") {
            return std::string("error: direction must be forward or back, got 'up'");
        }
        return std::string("cycle 1, pc 1, running");
    });

    const int press = connect_to(server.port());
    const std::string reply = send_request(press, "POST", "/move?direction=forward&unit=cycle", "running");
    ::close(press);
    CHECK(has(reply, "200 OK"));
    CHECK(has(reply, "cycle 1, pc 1, running"));
    CHECK_EQ(pressed, std::string("forward cycle"));

    const int bad = connect_to(server.port());
    CHECK(has(send_request(bad, "POST", "/move?direction=up&unit=cycle", "got 'up'"), "400 Bad Request"));
    ::close(bad);

    pressed = "";
    const int fetched = connect_to(server.port());
    CHECK(has(get(fetched, "/move?direction=forward&unit=cycle", "not found"), "404 Not Found"));   // a GET never moves it
    ::close(fetched);
    CHECK_EQ(pressed, std::string(""));
}

TEST(viz_server_passes_jumps_to_the_machine) {
    VizServer server(make_root(), 0);
    u64 asked = 99;
    server.on_jump([&asked](u64 cycle) {
        asked = cycle;
        return std::string("cycle 25, pc 2, running");
    });

    const int jump = connect_to(server.port());
    CHECK(has(send_request(jump, "POST", "/jump?cycle=25", "running"), "200 OK"));
    ::close(jump);
    CHECK_EQ(asked, u64{25});

    const int bad = connect_to(server.port());
    CHECK(has(send_request(bad, "POST", "/jump?cycle=x", "cycle=N"), "400 Bad Request"));
    ::close(bad);
}
