/// Runs every TEST, or only those whose name contains an argument: tpu_tests isa

#include <cstdio>
#include <cstring>

#include "test_framework.h"

namespace {

// With no arguments every test runs; otherwise a test runs if its name contains any argument.
bool selected(const char* name, int argc, char** argv) {
    if (argc == 1) {
        return true;
    }
    for (int i = 1; i < argc; ++i) {
        const bool name_contains_argument = std::strstr(name, argv[i]) != nullptr;
        if (name_contains_argument) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    int ran = 0;
    int failed_tests = 0;

    for (const test::Case& c : test::registry()) {
        if (!selected(c.name, argc, argv)) {
            continue;
        }

        const int failures_before = test::failed_checks;
        c.run();
        ran = ran + 1;

        const bool passed = test::failed_checks == failures_before;
        if (passed) {
            std::printf("pass %s\n", c.name);
        } else {
            std::printf("FAIL %s\n", c.name);
            failed_tests = failed_tests + 1;
        }
    }

    std::printf("\n%d tests, %d failed\n", ran, failed_tests);
    if (failed_tests > 0) {
        return 1;
    }
    return 0;
}
