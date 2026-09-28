/// The assembler: program text to a Program, with errors reported as name:line:col.

#pragma once

#include <stdexcept>
#include <string>

#include "isa/isa.h"

struct AsmError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// One statement per line: `Op key=value ...`, `.host ROW` or `.weights TILE` opening a data block, or int8 values; `#` comments.
Program assemble(const std::string& source, const std::string& name = "<input>");
Program assemble_file(const std::string& path);
