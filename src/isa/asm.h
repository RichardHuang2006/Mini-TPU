/// The assembler: program text to a Program, with errors reported as name:line:col.

#pragma once

#include <stdexcept>
#include <string>

#include "isa/isa.h"

struct AsmError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One statement per line: `Op key=value ...`, `.host ROW` or `.weights TILE [ROW]` opening a data block, or int8 values; `#` comments.
Program assemble(const std::string& source, const std::string& name = "<input>");
Program assemble_file(const std::string& path);

// The program as text assemble() reads back: a one-line header comment, its data by row (trailing zeros left out), then its code.
std::string program_text(const Program& program, const std::string& header);
