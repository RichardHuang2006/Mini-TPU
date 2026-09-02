// Test driver. Each test_*.cpp translation unit registers its SECTION()s in
// the shared registry (tests/test_support.h); this file only runs them all
// and reports the totals.

#include <cstdio>

#include "test_support.h"

int main() {
    // Line-buffered, so a section header, its assertion failures and anything a
    // test writes to stderr stay in the order they happened even when the output
    // is piped to a file or a CI log.
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
