/// Snapshot JSON: the fields the page draws, at known cycles of a copy program, and pinned targets.

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
    shell.execute("step 5");

    const std::string json = snapshot_json(tpu, shell);
    CHECK_EQ(json.front(), '{');
    CHECK_EQ(json.back(), '}');
    CHECK(json.find('\n') == std::string::npos);   // one line: an SSE event cannot hold a raw newline
    CHECK(has(json, "\"cycle\":5"));
    CHECK(has(json, "\"state\":\"stalled: host interface busy\""));
    CHECK(has(json, "\"blame\":\"host\""));
    CHECK(has(json, "\"done\":110,\"total\":512"));   // five cycles of 22 bytes
    CHECK(has(json, "\"issue\":\"Ihhhh\""));           // the read issued, then the write waited four cycles
    CHECK(has(json, "\"host\":\"#####\""));
    CHECK(has(json, "\"Read_Host_Memory host=0x0 ub=0x0 rows=2\""));
    CHECK(has(json, "\"nonzero\":\"#"));                // UB cell 0 now holds the first bytes
    CHECK(has(json, "\"age\":[1,"));                    // written during cycle 4, the one just finished
}

TEST(snapshot_keeps_pinned_targets_live) {
    Tpu tpu;
    Shell shell(tpu);
    shell.execute("load " + write_temp_program("mini_tpu_snapshot_pin.s", kCopy));
    shell.execute("ub[0:1x4]");

    CHECK(has(snapshot_json(tpu, shell), "\"target\":\"ub[0:1x4]\",\"text\":\"0x0:    0    0    0    0\""));
    shell.execute("run");
    CHECK(has(snapshot_json(tpu, shell), "\"target\":\"ub[0:1x4]\",\"text\":\"0x0:    1    2    3    4\""));
}

TEST(snapshot_tells_the_observer_after_every_command) {
    Tpu tpu;
    Shell shell(tpu);
    int calls = 0;
    shell.set_observer([&calls]() { calls = calls + 1; });

    shell.execute("load " + write_temp_program("mini_tpu_snapshot_obs.s", kCopy));
    shell.execute("step");
    shell.execute("bogus");
    CHECK_EQ(calls, 3);
}
