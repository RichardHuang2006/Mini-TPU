/// ISA: 12-byte layout, round trips at field limits, overflow and unknown-opcode errors, disassembly text.

#include <string>

#include "isa/isa.h"
#include "test_framework.h"

namespace {

Instr host_op(Op op, u32 ub_row, u32 host_row, u32 rows) {
    Instr in;
    in.op       = op;
    in.ub_row   = ub_row;
    in.host_row = host_row;
    in.rows     = rows;
    return in;
}

bool same(const Instr& a, const Instr& b) {
    return a.op == b.op && a.ub_row == b.ub_row && a.host_row == b.host_row && a.rows == b.rows;
}

}  // namespace

TEST(isa_host_transfer_byte_layout) {
    const InstrBytes b = encode(host_op(Op::ReadHostMemory, 0x030201, 0x07060504, 0x0908));
    const InstrBytes want = {0x10, 0, 0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    CHECK(b == want);
}

TEST(isa_round_trips_at_field_limits) {
    for (Op op : {Op::ReadHostMemory, Op::WriteHostMemory}) {
        const Instr max = host_op(op, 0xFFFFFF, 0xFFFFFFFF, 0xFFFF);
        const Instr min = host_op(op, 0, 0, 0);
        CHECK(same(decode(encode(max)), max));
        CHECK(same(decode(encode(min)), min));
    }
    for (Op op : {Op::Nop, Op::Halt, Op::Sync}) {
        Instr in;
        in.op = op;
        CHECK(same(decode(encode(in)), in));
        CHECK(encode(in) == InstrBytes{static_cast<u8>(op)});   // every other byte zero
    }
}

TEST(isa_fields_that_do_not_fit_throw) {
    CHECK_THROWS(encode(host_op(Op::ReadHostMemory, 1u << 24, 0, 1)));
    CHECK_THROWS(encode(host_op(Op::WriteHostMemory, 0, 0, 1u << 16)));
}

TEST(isa_unknown_opcode_throws) {
    CHECK_THROWS(decode(InstrBytes{0x03}));
    CHECK_THROWS(decode(InstrBytes{0xFF}));
}

TEST(isa_disassembly) {
    CHECK_EQ(disasm(host_op(Op::ReadHostMemory, 0x10, 0x200, 4)), std::string("Read_Host_Memory host=0x200 ub=0x10 rows=4"));
    CHECK_EQ(disasm(host_op(Op::WriteHostMemory, 0x10, 0x200, 4)), std::string("Write_Host_Memory ub=0x10 host=0x200 rows=4"));
    Instr halt;
    halt.op = Op::Halt;
    CHECK_EQ(disasm(halt), std::string("Halt"));
}
