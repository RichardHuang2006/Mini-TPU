/// The visualizer's local web server: serves the page, pushes every snapshot to it, and takes its button presses.

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/types.h"
#include "viz/state.h"

class VizServer {
public:
    // Listens on 127.0.0.1 at the first free port of first_port..first_port+9 (0 lets the system pick); throws if none is free.
    VizServer(std::string root, u32 first_port);

    // Stops listening and closes every page's connection.
    ~VizServer();

    VizServer(const VizServer&) = delete;
    VizServer& operator=(const VizServer&) = delete;

    u32 port() const;

    // Sends this snapshot to every open page, and keeps the state copy that /rows and /pes read from.
    void publish(const std::string& json, std::shared_ptr<const MachineState> state);

    // What POST /move?direction=D&unit=U calls; its text is the reply, sent as 400 when it starts with "error: ".
    using MoveHandler = std::function<std::string(const std::string& direction, const std::string& unit)>;
    void on_move(MoveHandler handler);

private:
    std::string root_;   // the folder holding index.html, style.css and app.js
    int         listen_fd_ = -1;
    u32         port_      = 0;

    std::atomic<bool>        stopping_{false};
    std::thread              accept_thread_;
    std::vector<std::thread> client_threads_;   // one per connection

    std::mutex              mutex_;   // guards latest_, state_, version_, move_ and client_threads_
    std::condition_variable changed_;
    std::string             latest_;
    std::shared_ptr<const MachineState> state_;
    u64                     version_ = 0;
    MoveHandler             move_;

    void accept_loop();
    void serve(int fd);
    void serve_file(int fd, const std::string& name, const std::string& type);
    void serve_events(int fd);
    void serve_data(int fd, const std::string& path, const std::string& query);
    void serve_move(int fd, const std::string& query);
};
