/// Snapshot JSON: the fields the page draws, at known cycles of a copy program, and the observer hook.

#include <filesystem>
#include <fstream>
#include <string>

#include "test_framework.h"
#include "viz/snapshot.h"

namespace {

const char* kCopy =
    ".host 0\n"
    "1 2 3 4\n"
    "Read_Host_Memory host=0 ub=0 rows=2\n"
    "Write_Host_Memory ub=0 host=4 rows=2\n"
    "Halt\n";

std::string write_temp_program(const std::string& name, const std::string& source) {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << source;
    return path.string();
}

bool has(const std::string& json, const std::string& piece) {
    return json.find(piece) != std::string::npos;
}

}  // namespace

TEST(snapshot_shows_the_machine_mid_copy) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_snapshot.s", kCopy));
    tpu.run_cycles(5);

    const std::string json = snapshot_json(tpu);
    CHECK_EQ(json.front(), '{');
    CHECK_EQ(json.back(), '}');
    CHECK(json.find('\n') == std::string::npos);   // one line: an SSE event cannot hold a raw newline
    CHECK(has(json, "\"cycle\":5"));
    CHECK(has(json, "\"state\":\"stalled: host interface busy\""));
    CHECK(has(json, "\"blame\":\"host\""));
    CHECK(has(json, "\"done\":110,\"total\":512"));   // five cycles of 22 bytes
    CHECK(has(json, "\"pages\":[0],\"written\":[[0,1]]"));   // host page 0, row 0 holds the program's data

    // The host block's "rows" is the transfer's row count, and appears once: a second key would replace it.
    const std::size_t host_start = json.find("\"host\":{");
    const std::string host = json.substr(host_start, json.find('}', host_start) - host_start);
    CHECK(has(host, "\"rows\":2,"));
    CHECK_EQ(host.find("\"rows\":"), host.rfind("\"rows\":"));
    CHECK(has(json, "\"Read_Host_Memory host=0x0 ub=0x0 rows=2\""));
    CHECK(has(json, "\"ub_rows\":[[0,1]]"));          // 110 bytes have landed, all in UB row 0
    CHECK(has(json, "\"acc_rows\":[]"));

    // Cycle 0 issued the read and started the host interface on it; cycle 1's write waits for it.
    CHECK(has(json, "\"timeline\":{\"first\":0,\"length\":64,\"cycles\":[{\"stall\":0,\"issued\":0,\"host\":0,"));
    CHECK(has(json, "{\"stall\":1,\"issued\":-1,\"host\":0,\"fetching\":-1,\"shifting\":-1,\"mxu\":-1,\"activation\":-1}"));
}

TEST(snapshot_tells_the_observer_after_every_line_and_press) {
    Tpu tpu;
    Shell shell(tpu);
    int calls = 0;
    shell.set_observer([&calls]() { calls = calls + 1; });

    shell.execute("load " + write_temp_program("mini_tpu_snapshot_obs.s", kCopy));
    shell.move("forward", "cycle");
    shell.execute("bogus");
    CHECK_EQ(calls, 3);
}
