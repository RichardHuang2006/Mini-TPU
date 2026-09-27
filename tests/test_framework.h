#pragma once

// A dependency-free test harness.
//
//   TEST(bf16_rounds_ties_to_even) {
//       CHECK(x == y);
//       CHECK_EQ(bf16::from_f32(1.0f).bits, 0x3F80);
//       CHECK_THROWS(mem.read(1 << 30));
//   }
//
// A failed check reports and lets the test continue, so one run shows every
// broken expectation, not just the first.

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* name;
    void      (*fn)();
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int failed_checks = 0;   // across the whole run

inline bool add(const char* name, void (*fn)()) {
    registry().push_back({name, fn});
    return true;
}

inline void fail(const char* file, int line, const std::string& what) {
    ++failed_checks;
    std::printf("  FAIL %s:%d: %s\n", file, line, what.c_str());
}

template <typename A, typename B>
void check_eq(const A& a, const B& b, const char* a_src, const char* b_src,
              const char* file, int line) {
    if (a == b) return;
    std::ostringstream os;
    os << a_src << " == " << b_src << "  (" << a << " vs " << b << ")";
    fail(file, line, os.str());
}

}  // namespace test

#define TEST(name)                                                   \
    static void name();                                              \
    static const bool name##_registered = test::add(#name, name);    \
    static void name()

#define CHECK(cond)                                                  \
    do {                                                             \
        if (!(cond)) test::fail(__FILE__, __LINE__, #cond);          \
    } while (0)

#define CHECK_EQ(a, b) test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)

#define CHECK_THROWS(expr)                                           \
    do {                                                             \
        bool threw_ = false;                                         \
        try { (void)(expr); } catch (...) { threw_ = true; }         \
        if (!threw_) test::fail(__FILE__, __LINE__, #expr " did not throw"); \
    } while (0)
