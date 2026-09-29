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

Instr no_operands(Op op) {
    Instr in;
    in.op = op;
    return in;
}

bool same(const Instr& a, const Instr& b) {
    const bool same_op     = a.op == b.op;
    const bool same_fields = a.ub_row == b.ub_row && a.host_row == b.host_row && a.rows == b.rows;
    return same_op && same_fields;
}

}  // namespace

TEST(isa_host_transfer_byte_layout) {
    // Each field's bytes are numbered so their positions are easy to see: ub 01 02 03, host 04..07, rows 08 09.
    const Instr in = host_op(Op::ReadHostMemory, 0x030201, 0x07060504, 0x0908);
    const InstrBytes expected = {0x10, 0, 0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    CHECK(encode(in) == expected);
}

TEST(isa_read_weights_byte_layout) {
    Instr in;
    in.op   = Op::ReadWeights;
    in.tile = 0x04030201;
    const InstrBytes expected = {0x20, 0, 0, 0x01, 0x02, 0x03, 0x04, 0, 0, 0, 0, 0};
    CHECK(encode(in) == expected);

    in.tile = 0xFFFFFFFF;
    CHECK_EQ(decode(encode(in)).tile, u32{0xFFFFFFFF});
    CHECK_EQ(disasm(in), std::string("Read_Weights tile=0xFFFFFFFF"));
}

TEST(isa_matrix_multiply_byte_layout) {
    Instr in;
    in.op          = Op::MatrixMultiply;
    in.ub_row      = 0x030201;
    in.acc_row     = 0x0504;
    in.rows        = 0x09080706;
    in.accumulate  = 1;
    in.new_weights = 1;
    const InstrBytes expected = {0x30, 0x03, 0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09};
    CHECK(encode(in) == expected);

    const Instr back = decode(encode(in));
    CHECK_EQ(back.acc_row, u32{0x0504});
    CHECK_EQ(back.rows, u32{0x09080706});
    CHECK_EQ(back.accumulate, u32{1});
    CHECK_EQ(back.new_weights, u32{1});
    CHECK_EQ(disasm(in), std::string("MatrixMultiply ub=0x30201 acc=0x504 rows=151521030 accumulate=1 new_weights=1"));

    in.accumulate = 2;   // flags are 0 or 1
    CHECK_THROWS(encode(in));
}

TEST(isa_round_trips_at_field_limits) {
    for (Op op : {Op::ReadHostMemory, Op::WriteHostMemory}) {
        const Instr largest  = host_op(op, 0xFFFFFF, 0xFFFFFFFF, 0xFFFF);
        const Instr smallest = host_op(op, 0, 0, 0);
        CHECK(same(decode(encode(largest)), largest));
        CHECK(same(decode(encode(smallest)), smallest));
    }

    for (Op op : {Op::Nop, Op::Halt, Op::Sync}) {
        const Instr in = no_operands(op);
        CHECK(same(decode(encode(in)), in));

        InstrBytes only_opcode{};   // the opcode, then eleven zero bytes
        only_opcode[0] = static_cast<u8>(op);
        CHECK(encode(in) == only_opcode);
    }
}

TEST(isa_fields_that_do_not_fit_throw) {
    const u32 ub_needs_25_bits   = 1u << 24;
    const u32 rows_needs_17_bits = 1u << 16;
    CHECK_THROWS(encode(host_op(Op::ReadHostMemory, ub_needs_25_bits, 0, 1)));
    CHECK_THROWS(encode(host_op(Op::WriteHostMemory, 0, 0, rows_needs_17_bits)));
}

TEST(isa_unknown_opcode_throws) {
    InstrBytes bytes{};
    bytes[0] = 0x03;
    CHECK_THROWS(decode(bytes));
    bytes[0] = 0xFF;
    CHECK_THROWS(decode(bytes));
}

TEST(isa_disassembly) {
    const Instr read  = host_op(Op::ReadHostMemory, 0x10, 0x200, 4);
    const Instr write = host_op(Op::WriteHostMemory, 0x10, 0x200, 4);

    CHECK_EQ(disasm(read), std::string("Read_Host_Memory host=0x200 ub=0x10 rows=4"));
    CHECK_EQ(disasm(write), std::string("Write_Host_Memory ub=0x10 host=0x200 rows=4"));
    CHECK_EQ(disasm(no_operands(Op::Halt)), std::string("Halt"));
}
