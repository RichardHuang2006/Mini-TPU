/// Dependency-free test harness; a failed check reports and the test keeps going.

#pragma once

#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* name;
    void (*run)();
};

// Every TEST adds itself here before main() starts.
inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int failed_checks = 0;   // across the whole run

inline bool add(const char* name, void (*run)()) {
    registry().push_back({name, run});
    return true;
}

inline void fail(const char* file, int line, const std::string& what) {
    failed_checks = failed_checks + 1;
    std::printf("  FAIL %s:%d: %s\n", file, line, what.c_str());
}

// Reports both values when they differ, e.g. "x == 3  (4 vs 3)".
template <typename A, typename B>
void check_eq(const A& a, const B& b, const char* a_text, const char* b_text, const char* file, int line) {
    if (a == b) {
        return;
    }
    std::ostringstream message;
    message << a_text << " == " << b_text << "  (" << a << " vs " << b << ")";
    fail(file, line, message.str());
}

}  // namespace test

// TEST(name) { ... } declares a function and registers it, via a static bool set before main().
#define TEST(name)                                                    \
    static void name();                                               \
    static const bool name##_registered = test::add(#name, name);     \
    static void name()

#define CHECK(condition)                                              \
    do {                                                              \
        if (!(condition)) {                                           \
            test::fail(__FILE__, __LINE__, #condition);               \
        }                                                             \
    } while (0)

#define CHECK_EQ(a, b) test::check_eq((a), (b), #a, #b, __FILE__, __LINE__)

#define CHECK_THROWS(expression)                                          \
    do {                                                                  \
        bool threw = false;                                               \
        try {                                                             \
            (void)(expression);                                           \
        } catch (...) {                                                   \
            threw = true;                                                 \
        }                                                                 \
        if (!threw) {                                                     \
            test::fail(__FILE__, __LINE__, #expression " did not throw"); \
        }                                                                 \
    } while (0)
