/// The machine: load, in-order issue, exact cycle counts and stall causes, step and step back, errors at the PC.

#include <stdexcept>
#include <string>
#include <vector>

#include "core/tpu.h"
#include "isa/asm.h"
#include "test_framework.h"

namespace {

// Host row 0 holds 1, 2, 3, 4; one Read_Host_Memory moves that row to UB row 5.
const char* kReadThenHalt =
    ".host 0\n"
    "1 2 3 4\n"
    "Read_Host_Memory host=0 ub=5 rows=1\n"
    "Halt\n";

// Two reads in a row: the second waits for the host interface.
const char* kTwoReads =
    "Read_Host_Memory host=0 ub=0 rows=1\n"
    "Read_Host_Memory host=1 ub=1 rows=1\n"
    "Halt\n";

std::string error_of_run(Tpu& tpu) {
    try {
        tpu.run_to_halt();
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(tpu_load_places_data_and_resets) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));

    i8 byte = 0;
    tpu.host().read(3, &byte, 1);
    CHECK_EQ(byte, 4);
    CHECK_EQ(tpu.cycle(), Cycle{0});
    CHECK_EQ(tpu.pc(), u32{0});
    CHECK(!tpu.halted());
}

TEST(tpu_halt_waits_for_the_transfer) {
    Tpu tpu;
    tpu.load(assemble(kReadThenHalt));
    tpu.run_to_halt();

    // The read issues on cycle 0 and moves bytes on cycles 0-11; Halt waits 11 cycles, then issues on cycle 12.
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.stats().stalled(Stall::WaitForIdle), u64{11});
    CHECK_EQ(tpu.pc(), u32{1});   // Halt keeps the PC on itself
    CHECK_EQ(tpu.ub().row(5)[3], 4);
}

TEST(tpu_second_transfer_waits_for_the_first) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));

    tpu.run(5);
    CHECK(tpu.stall() == Stall::HostInterfaceBusy);

    tpu.run_to_halt();
    CHECK_EQ(tpu.cycle(), Cycle{25});
    CHECK_EQ(tpu.stats().stalled(Stall::HostInterfaceBusy), u64{11});
    CHECK_EQ(tpu.stats().stalled(Stall::WaitForIdle), u64{11});
}

TEST(tpu_round_trip_through_the_ub) {
    const char* source =
        ".host 0\n"
        "-5 6 -7\n"
        "Read_Host_Memory host=0 ub=0 rows=1\n"
        "Write_Host_Memory ub=0 host=7 rows=1\n"
        "Halt\n";
    Tpu tpu;
    tpu.load(assemble(source));
    tpu.run_to_halt();

    std::vector<i8> back(3, 0);
    tpu.host().read(7 * 256, back.data(), back.size());
    CHECK(back == std::vector<i8>({-5, 6, -7}));
}

TEST(tpu_sync_waits_for_idle) {
    Tpu tpu;
    tpu.load(assemble("Read_Host_Memory host=0 ub=0 rows=1\nSync\nNop\nHalt\n"));

    tpu.step(2);   // the read, then Sync once the read is done
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK(!tpu.host_interface().busy());
}

TEST(tpu_step_stops_right_after_an_issue) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));

    tpu.step(1);
    CHECK_EQ(tpu.issued(), u64{1});
    CHECK_EQ(tpu.cycle(), Cycle{1});

    tpu.step(1);   // skips the 11 stalled cycles
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK(tpu.stall() == Stall::None);
}

TEST(tpu_step_back_replays_to_the_same_state) {
    Tpu tpu;
    tpu.load(assemble(kTwoReads));
    tpu.run_to_halt();
    CHECK_EQ(tpu.cycle(), Cycle{25});

    tpu.step_back(1);   // undo Halt: back to just after the second read issued
    CHECK_EQ(tpu.issued(), u64{2});
    CHECK_EQ(tpu.cycle(), Cycle{13});
    CHECK_EQ(tpu.pc(), u32{2});
    CHECK(!tpu.halted());
    CHECK_EQ(tpu.host_interface().bytes_done(), u64{22});   // the second transfer's first cycle

    tpu.step(1);   // forward again lands where it was
    CHECK(tpu.halted());
    CHECK_EQ(tpu.cycle(), Cycle{25});

    tpu.step_back(10);   // more than were issued: back to the start
    CHECK_EQ(tpu.issued(), u64{0});
    CHECK_EQ(tpu.cycle(), Cycle{0});
}

TEST(tpu_errors_name_the_pc_and_leave_the_cycle_alone) {
    Tpu tpu;
    tpu.load(assemble("Read_Host_Memory host=0 ub=0x20000 rows=1\nHalt\n"));   // UB row 131072 is past the last
    const std::string message = error_of_run(tpu);
    const std::string expected_start = "pc 0: Read_Host_Memory host=0x0 ub=0x20000 rows=1: ub bytes";
    CHECK_EQ(message.substr(0, expected_start.size()), expected_start);
    CHECK_EQ(tpu.cycle(), Cycle{0});
    CHECK_EQ(tpu.issued(), u64{0});

    tpu.load(assemble("Nop\n"));
    CHECK_EQ(error_of_run(tpu), std::string("pc 1: past the end of the program; is a Halt missing?"));

    Program unknown;
    InstrBytes bad{};
    bad[0] = 0x03;
    unknown.code.push_back(bad);
    tpu.load(unknown);
    CHECK_EQ(error_of_run(tpu), std::string("pc 0: unknown opcode 0x3"));
}
