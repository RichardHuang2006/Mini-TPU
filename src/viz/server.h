/// The visualizer's local web server: serves the page from viz/ and pushes every snapshot to it as a Server-Sent Event.

#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/types.h"

class VizServer {
public:
    // Listens on 127.0.0.1 at the first free port of first_port..first_port+9 (0 lets the system pick); throws if none is free.
    VizServer(std::string root, u32 first_port);

    // Stops listening and closes every page's connection.
    ~VizServer();

    VizServer(const VizServer&) = delete;
    VizServer& operator=(const VizServer&) = delete;

    u32 port() const;

    // Sends this snapshot to every open page; a page that connects later gets it too.
    void publish(const std::string& json);

private:
    std::string root_;   // the folder holding index.html, style.css and app.js
    int         listen_fd_ = -1;
    u32         port_      = 0;

    std::atomic<bool>        stopping_{false};
    std::thread              accept_thread_;
    std::vector<std::thread> client_threads_;   // one per connection

    std::mutex              mutex_;   // guards latest_, version_ and client_threads_
    std::condition_variable changed_;
    std::string             latest_;
    u64                     version_ = 0;

    void accept_loop();
    void serve(int fd);
    void serve_file(int fd, const std::string& name, const std::string& type);
    void serve_events(int fd);
};
