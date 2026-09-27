/// The tiny() and v4() presets: their values and the invariants the machine relies on.

#include <string>

#include "common/config.h"
#include "test_framework.h"

namespace {

bool pow2(u32 x) { return x != 0 && (x & (x - 1)) == 0; }

void check_invariants(const CoreConfig& c) {
    CHECK_EQ(c.lanes, c.mxu_dim);                     // one vreg row is one MXU input row
    CHECK(c.num_mxus >= 1);
    CHECK(c.dma_words_per_cycle >= 1);
    CHECK(pow2(c.main_words) && pow2(c.vmem_words) && pow2(c.smem_words));
    CHECK(c.vmem_words >= 4 * c.mxu_dim * c.mxu_dim);  // room for A, B and C tiles plus a spare
    CHECK(c.main_words >= c.vmem_words);
}

}  // namespace

TEST(config_tiny_preset) {
    const CoreConfig c = CoreConfig::tiny();
    CHECK_EQ(std::string(c.name), "tiny");
    CHECK_EQ(c.mxu_dim, 8u);
    CHECK_EQ(c.num_mxus, 1u);
    CHECK_EQ(c.vreg_words(), 64u);
    check_invariants(c);
}

TEST(config_v4_preset) {
    const CoreConfig c = CoreConfig::v4();
    CHECK_EQ(std::string(c.name), "v4");
    CHECK_EQ(c.mxu_dim, 128u);
    CHECK_EQ(c.num_mxus, 4u);
    CHECK_EQ(c.vreg_words(), 1024u);
    check_invariants(c);
}
