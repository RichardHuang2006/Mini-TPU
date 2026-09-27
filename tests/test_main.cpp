/// Runs every TEST, or only those whose name contains an argument: tpu_tests bf16

#include <cstdio>
#include <cstring>

#include "test_framework.h"

int main(int argc, char** argv) {
    int ran = 0;
    int failed_tests = 0;

    for (const test::Case& c : test::registry()) {
        bool selected = argc == 1;
        for (int i = 1; i < argc && !selected; ++i)
            selected = std::strstr(c.name, argv[i]) != nullptr;
        if (!selected) continue;

        const int before = test::failed_checks;
        c.fn();
        ++ran;
        const bool ok = test::failed_checks == before;
        if (!ok) ++failed_tests;
        std::printf("%s %s\n", ok ? "pass" : "FAIL", c.name);
    }

    std::printf("\n%d test%s, %d failed\n", ran, ran == 1 ? "" : "s", failed_tests);
    return failed_tests == 0 ? 0 : 1;
}
