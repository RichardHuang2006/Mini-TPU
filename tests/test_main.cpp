// Runs every SECTION() the test_*.cpp files registered in tests/test_support.h.

#include <cstdio>

#include "test_support.h"

int main() {
    // Line-buffered, so stdout and stderr stay interleaved in order when piped.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    int passes = 0, fails = 0;
    for (auto& [name, fn] : test::registry()) {
        const int before = test::assertion_failures;
        std::printf("=== %s ===\n", name.c_str());
        fn();
        if (test::assertion_failures > before) ++fails;
        else                                   ++passes;
    }
    std::printf("\n%d section%s ok, %d failing (%d assertion failure%s)\n",
                passes, passes == 1 ? "" : "s",
                fails,
                test::assertion_failures, test::assertion_failures == 1 ? "" : "s");
    return test::assertion_failures ? 1 : 0;
}
