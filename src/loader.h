#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <istream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "isa.h"
#include "tensor.h"
#include "types.h"

// Program and tensor input. Each entry point takes a stream so tests can drive
// it from a string and the CLI from a file.
//
// Three tensor formats: the MTPU container carries its own shape, so a bundled
// example is self-describing, while raw and hex do not, so a tensor dumped by an
// external framework can be fed in with the shape given on the command line. All
// three must land the same bytes.

namespace loader_detail {

inline std::vector<uint8_t> read_all_bytes(std::istream& in) {
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)),
                                std::istreambuf_iterator<char>());
}

// One hex value per line. `#` and `//` start a comment, blank lines are
// skipped, an optional 0x prefix is accepted.
inline std::vector<uint32_t> read_hex_values(std::istream& in, const char* who) {
    std::vector<uint32_t> out;
    std::string line;
    while (std::getline(in, line)) {
        // Strip '\r' first, or it hides behind an end-of-line comment.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (auto p = line.find("//"); p != std::string::npos) line.erase(p);
        if (auto p = line.find('#');  p != std::string::npos) line.erase(p);
        auto is_ws = [](char c) { return c == ' ' || c == '\t'; };
        while (!line.empty() && is_ws(line.front())) line.erase(0, 1);
        while (!line.empty() && is_ws(line.back()))  line.pop_back();
        if (line.empty()) continue;

        if (line.size() >= 2 && line[0] == '0' && (line[1] == 'x' || line[1] == 'X')) {
            line.erase(0, 2);
        }
        if (line.empty() || line.find_first_not_of("0123456789abcdefABCDEF")
                                != std::string::npos) {
            throw std::runtime_error(std::string(who) + ": invalid hex value '" + line + "'");
        }
        if (line.size() > 8) {
            throw std::runtime_error(std::string(who) + ": hex value wider than 32 bits '" + line + "'");
        }
        out.push_back(static_cast<uint32_t>(std::stoul(line, nullptr, 16)));
    }
    return out;
}

inline uint32_t rd_u32_le(const uint8_t* p) {
    return  static_cast<uint32_t>(p[0])        |
           (static_cast<uint32_t>(p[1]) <<  8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline void wr_u32_le(std::ostream& out, uint32_t v) {
    const char b[4] = {static_cast<char>(v & 0xFF), static_cast<char>((v >> 8) & 0xFF),
                       static_cast<char>((v >> 16) & 0xFF), static_cast<char>((v >> 24) & 0xFF)};
    out.write(b, 4);
}

}  // namespace loader_detail

// ---- programs -------------------------------------------------------------

// One hex word per line, ISA_WORDS consecutive words per instruction.
inline std::vector<RawInst> load_program_hex(std::istream& in) {
    const std::vector<uint32_t> words =
        loader_detail::read_hex_values(in, "load_program_hex");
    if (words.size() % ISA_WORDS != 0) {
        throw std::runtime_error("load_program_hex: word count is not a multiple of "
                                 + std::to_string(ISA_WORDS));
    }
    std::vector<RawInst> prog(words.size() / ISA_WORDS);
    for (std::size_t i = 0; i < prog.size(); ++i) {
        for (uint32_t w = 0; w < ISA_WORDS; ++w) {
            prog[i].word[w] = words[i * ISA_WORDS + w];
        }
    }
    return prog;
}

// Little-endian binary image of the same words.
inline std::vector<RawInst> load_program_raw(std::istream& in) {
    const std::vector<uint8_t> bytes = loader_detail::read_all_bytes(in);
    if (bytes.size() % ISA_INST_BYTES != 0) {
        throw std::runtime_error("load_program_raw: size is not a multiple of "
                                 + std::to_string(ISA_INST_BYTES));
    }
    std::vector<RawInst> prog(bytes.size() / ISA_INST_BYTES);
    for (std::size_t i = 0; i < prog.size(); ++i) {
        for (uint32_t w = 0; w < ISA_WORDS; ++w) {
            prog[i].word[w] = loader_detail::rd_u32_le(
                &bytes[i * ISA_INST_BYTES + w * 4]);
        }
    }
    return prog;
}

inline void save_program_raw(std::ostream& out, const std::vector<RawInst>& prog) {
    for (const RawInst& inst : prog) {
        for (uint32_t w = 0; w < ISA_WORDS; ++w) {
            loader_detail::wr_u32_le(out, inst.word[w]);
        }
    }
}

inline void save_program_hex(std::ostream& out, const std::vector<RawInst>& prog) {
    char buf[16];
    for (const RawInst& inst : prog) {
        for (uint32_t w = 0; w < ISA_WORDS; ++w) {
            std::snprintf(buf, sizeof buf, "%08X\n", inst.word[w]);
            out << buf;
        }
    }
}

// ---- tensors --------------------------------------------------------------

// A loaded tensor. `wide` selects which vector is populated: activations and
// weights are int8, a dumped accumulator is int32.
struct TensorBlob {
    uint32_t rows = 0;
    uint32_t cols = 0;
    bool     wide = false;

    std::vector<i8>  i8v;
    std::vector<i32> i32v;

    std::size_t count() const { return static_cast<std::size_t>(rows) * cols; }

    ConstI8View  i8_view()  const { return ConstI8View(i8v.data(), rows, cols); }
    ConstI32View i32_view() const { return ConstI32View(i32v.data(), rows, cols); }
    I8View       i8_view()        { return I8View(i8v.data(), rows, cols); }
    I32View      i32_view()       { return I32View(i32v.data(), rows, cols); }
};

// Self-describing container: "MTPU", version, dtype, rows, cols, then elements
// little-endian.
inline constexpr uint32_t MTPU_VERSION = 1;

inline void save_tensor_mtpu(std::ostream& out, const TensorBlob& t) {
    out.write("MTPU", 4);
    loader_detail::wr_u32_le(out, MTPU_VERSION);
    loader_detail::wr_u32_le(out, t.wide ? 1u : 0u);
    loader_detail::wr_u32_le(out, t.rows);
    loader_detail::wr_u32_le(out, t.cols);
    if (t.wide) {
        for (const i32 v : t.i32v) loader_detail::wr_u32_le(out, static_cast<uint32_t>(v));
    } else {
        for (const i8 v : t.i8v) out.put(static_cast<char>(v));
    }
}

inline TensorBlob load_tensor_mtpu(std::istream& in) {
    const std::vector<uint8_t> bytes = loader_detail::read_all_bytes(in);
    constexpr std::size_t HDR = 20;
    if (bytes.size() < HDR) {
        throw std::runtime_error("load_tensor_mtpu: file too small for a header");
    }
    if (bytes[0] != 'M' || bytes[1] != 'T' || bytes[2] != 'P' || bytes[3] != 'U') {
        throw std::runtime_error("load_tensor_mtpu: bad magic");
    }
    const uint32_t version = loader_detail::rd_u32_le(&bytes[4]);
    if (version != MTPU_VERSION) {
        throw std::runtime_error("load_tensor_mtpu: unsupported version "
                                 + std::to_string(version));
    }
    TensorBlob t;
    t.wide = loader_detail::rd_u32_le(&bytes[8]) != 0;
    t.rows = loader_detail::rd_u32_le(&bytes[12]);
    t.cols = loader_detail::rd_u32_le(&bytes[16]);

    const std::size_t elems = t.count();
    const std::size_t need  = HDR + elems * (t.wide ? 4u : 1u);
    if (bytes.size() < need) {
        throw std::runtime_error("load_tensor_mtpu: payload shorter than "
                                 + std::to_string(t.rows) + "x" + std::to_string(t.cols));
    }
    if (t.wide) {
        t.i32v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) {
            t.i32v[i] = static_cast<i32>(loader_detail::rd_u32_le(&bytes[HDR + i * 4]));
        }
    } else {
        t.i8v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) {
            t.i8v[i] = static_cast<i8>(bytes[HDR + i]);
        }
    }
    return t;
}

// Shape comes from the caller: a raw dump carries no metadata.
inline TensorBlob load_tensor_raw(std::istream& in, uint32_t rows, uint32_t cols, bool wide) {
    const std::vector<uint8_t> bytes = loader_detail::read_all_bytes(in);
    TensorBlob t;
    t.rows = rows;
    t.cols = cols;
    t.wide = wide;
    const std::size_t elems = t.count();
    if (bytes.size() < elems * (wide ? 4u : 1u)) {
        throw std::runtime_error("load_tensor_raw: file shorter than "
                                 + std::to_string(rows) + "x" + std::to_string(cols));
    }
    if (wide) {
        t.i32v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) {
            t.i32v[i] = static_cast<i32>(loader_detail::rd_u32_le(&bytes[i * 4]));
        }
    } else {
        t.i8v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) t.i8v[i] = static_cast<i8>(bytes[i]);
    }
    return t;
}

// One hex element per line; the low 8 bits are taken for an int8 tensor.
inline TensorBlob load_tensor_hex(std::istream& in, uint32_t rows, uint32_t cols, bool wide) {
    const std::vector<uint32_t> vals =
        loader_detail::read_hex_values(in, "load_tensor_hex");
    TensorBlob t;
    t.rows = rows;
    t.cols = cols;
    t.wide = wide;
    const std::size_t elems = t.count();
    if (vals.size() < elems) {
        throw std::runtime_error("load_tensor_hex: only " + std::to_string(vals.size())
                                 + " values for " + std::to_string(rows) + "x"
                                 + std::to_string(cols));
    }
    if (wide) {
        t.i32v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) t.i32v[i] = static_cast<i32>(vals[i]);
    } else {
        t.i8v.resize(elems);
        for (std::size_t i = 0; i < elems; ++i) {
            t.i8v[i] = static_cast<i8>(static_cast<uint8_t>(vals[i] & 0xFFu));
        }
    }
    return t;
}
