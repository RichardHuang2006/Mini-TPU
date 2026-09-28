/// Assembler: instructions, operand order, comments, data blocks, disasm round trip, and every error's position.

#include <string>
#include <vector>

#include "common/config.h"
#include "isa/asm.h"
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

Instr read_weights(u32 tile) {
    Instr in;
    in.op   = Op::ReadWeights;
    in.tile = tile;
    return in;
}

Instr no_operands(Op op) {
    Instr in;
    in.op = op;
    return in;
}

// The error message assembling `source` produces, or "no error".
std::string error_of(const std::string& source) {
    try {
        assemble(source, "t.s");
    } catch (const AsmError& e) {
        return e.what();
    }
    return "no error";
}

}  // namespace

TEST(asm_instructions_in_any_operand_order) {
    const Program p = assemble("Read_Host_Memory rows=4 ub=0x10 host=2\nSync\nHalt\n");

    CHECK_EQ(p.code.size(), std::size_t{3});
    if (p.code.size() == 3) {
        CHECK(p.code[0] == encode(host_op(Op::ReadHostMemory, 0x10, 2, 4)));
        CHECK(p.code[1] == encode(no_operands(Op::Sync)));
        CHECK(p.code[2] == encode(no_operands(Op::Halt)));
    }
}

TEST(asm_skips_comments_and_blank_lines) {
    const Program p = assemble("# a program\n\n   Nop   # trailing comment\n\t\n");
    CHECK_EQ(p.code.size(), std::size_t{1});
}

TEST(asm_data_blocks) {
    const std::string source =
        ".host 2\n"
        "1 -2 0x7F\n"
        "0xFF 0X80 -128\n"
        ".weights 1\n"
        "5\n"
        "Halt\n";
    const Program p = assemble(source);

    CHECK_EQ(p.code.size(), std::size_t{1});
    CHECK_EQ(p.host.size(), std::size_t{1});
    CHECK_EQ(p.weights.size(), std::size_t{1});
    if (p.host.size() != 1 || p.weights.size() != 1) {
        return;
    }

    const std::vector<i8> host_bytes = {1, -2, 127, -1, -128, -128};   // 0xFF is -1, 0X80 is -128
    CHECK_EQ(p.host[0].addr, 2 * u64{v1::kMxuDim});   // host memory is addressed in 256-byte rows
    CHECK(p.host[0].bytes == host_bytes);

    const std::vector<i8> weight_bytes = {5};
    CHECK_EQ(p.weights[0].addr, v1::kTileBytes);       // Weight Memory is addressed in 64 KiB tiles
    CHECK(p.weights[0].bytes == weight_bytes);
}

TEST(asm_reads_what_disasm_prints) {
    const std::vector<Instr> samples = {
        host_op(Op::ReadHostMemory, 0xFFFFFF, 0xFFFFFFFF, 0xFFFF),
        host_op(Op::WriteHostMemory, 0x10, 0x200, 4),
        no_operands(Op::Nop),
        no_operands(Op::Halt),
        no_operands(Op::Sync),
        read_weights(0x5),
    };

    for (const Instr& in : samples) {
        const Program p = assemble(disasm(in));
        CHECK_EQ(p.code.size(), std::size_t{1});
        if (p.code.size() == 1) {
            CHECK(p.code[0] == encode(in));
        }
    }
}

TEST(asm_errors_name_line_and_column) {
    CHECK_EQ(error_of("Nop\n  Read_Host_Mem ub=1"),
             std::string("t.s:2:3: error: unknown instruction 'Read_Host_Mem'"));
    CHECK_EQ(error_of("Read_Host_Memory hots=1 ub=1 rows=1"),
             std::string("t.s:1:18: error: Read_Host_Memory has no operand 'hots' (expects host, ub, rows)"));
    CHECK_EQ(error_of("Read_Host_Memory host=1 ub=1"),
             std::string("t.s:1:1: error: Read_Host_Memory is missing operand 'rows'"));
    CHECK_EQ(error_of("Read_Host_Memory host=1 host=2 ub=1 rows=1"),
             std::string("t.s:1:25: error: operand 'host' given twice"));
    CHECK_EQ(error_of("Read_Host_Memory host=zz ub=1 rows=1"),
             std::string("t.s:1:23: error: 'zz' is not a number"));
    CHECK_EQ(error_of("Read_Host_Memory host ub=1 rows=1"),
             std::string("t.s:1:18: error: expected key=value, got 'host'"));
    CHECK_EQ(error_of("Read_Host_Memory host=1 ub=1 rows=65536"),
             std::string("t.s:1:1: error: rows=65536 does not fit in 16 bits"));
    CHECK_EQ(error_of("Halt code=1"),
             std::string("t.s:1:6: error: Halt has no operand 'code' (it takes none)"));
    CHECK_EQ(error_of("1 2 3"),
             std::string("t.s:1:1: error: data outside a .host or .weights block"));
    CHECK_EQ(error_of(".host 0\n1 128"),
             std::string("t.s:2:3: error: value 128 is not an int8 (-128..127 or 0x00..0xFF)"));
    CHECK_EQ(error_of(".host 0\nHalt\n1"),
             std::string("t.s:3:1: error: data outside a .host or .weights block"));
    CHECK_EQ(error_of(".data 0"),
             std::string("t.s:1:1: error: unknown directive '.data' (expects .host or .weights)"));
    CHECK_EQ(error_of("Read_Weights"),
             std::string("t.s:1:1: error: Read_Weights is missing operand 'tile'"));
    CHECK_EQ(error_of(".host"),
             std::string("t.s:1:1: error: .host takes one address"));
}

TEST(asm_missing_file_throws) {
    CHECK_THROWS(assemble_file("/nonexistent/prog.s"));
}
