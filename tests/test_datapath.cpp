// Tests for src/datapath.h and src/config.h.

#include "test_support.h"

SECTION("types") {
    static_assert(std::is_same_v<i8,  int8_t>);
    static_assert(std::is_same_v<i32, int32_t>);
    static_assert(std::is_same_v<UbAddr,   uint32_t>);
    static_assert(std::is_same_v<HostAddr, uint32_t>);
    static_assert(std::is_same_v<BankId,   uint32_t>);
    static_assert(std::is_same_v<TileId,   uint32_t>);

    // Sentinels sit outside any plausible in-band value.
    REQUIRE(INVALID_UBADDR   > (1u << 24));
    REQUIRE(INVALID_HOSTADDR > (1u << 24));
    REQUIRE(INVALID_BANK     > (1u << 16));
    REQUIRE(INVALID_TILE     > (1u << 16));

    // Round-trip through the std::optional idiom the scoreboard will use.
    auto to_opt = [](BankId b) -> std::optional<BankId> {
        return b == INVALID_BANK ? std::nullopt : std::optional<BankId>(b);
    };
    auto from_opt = [](std::optional<BankId> o) -> BankId { return o.value_or(INVALID_BANK); };

    REQUIRE(!to_opt(INVALID_BANK).has_value());
    REQUIRE(from_opt(to_opt(INVALID_BANK)) == INVALID_BANK);
    for (const BankId b : std::initializer_list<BankId>{0u, 1u, 7u}) {
        REQUIRE(to_opt(b).has_value());
        REQUIRE(from_opt(to_opt(b)) == b);
    }

    // Enumerators are distinct.
    static_assert(static_cast<int>(Op::READ_HOST) != static_cast<int>(Op::HALT));
    static_assert(static_cast<int>(Op::MATMUL)    != static_cast<int>(Op::ACTIVATE));
    static_assert(static_cast<int>(ActFn::RELU)   != static_cast<int>(ActFn::RELU6));
    static_assert(static_cast<int>(Pool::MAX)     != static_cast<int>(Pool::AVG));

    // HALT is the last opcode, which the decoder's illegal-encoding check needs.
    static_assert(static_cast<uint32_t>(Op::HALT) == isa::MAX_OPCODE);
}

SECTION("config") {
    Config c;

    // Documented defaults.
    REQUIRE(c.dim                 == 32);
    REQUIRE(c.ub_bytes            == 256u * 1024u);
    REQUIRE(c.ub_banks            == 8);
    REQUIRE(c.acc_banks           == 4);
    REQUIRE(c.weight_fifo_depth   == 4);
    REQUIRE(c.double_buffer       == true);
    REQUIRE(c.dma_bytes_per_cycle == 16);

    // 1024 MACs per cycle, so the ridge point is 1024/16 = 64 MACs per byte.
    REQUIRE(c.peak_macs_per_cycle() == 1024u);
    REQUIRE(c.dma_bound(1000, 100));        // 10 MACs/byte: memory bound
    REQUIRE(!c.dma_bound(12800, 100));      // 128 MACs/byte: compute bound
    REQUIRE(!c.dma_bound(6400, 100));       // exactly at the ridge is not below it
    REQUIRE(c.dma_bound(6399, 100));        // one MAC under, and it tips over

    // A bigger array moves the ridge point up.
    Config big = c;
    big.dim = 256;
    REQUIRE(big.peak_macs_per_cycle() == 65536u);
    REQUIRE(big.dma_bound(12800, 100));

    static_assert(std::is_trivially_copyable_v<Config>);
}

SECTION("tensor") {
    // A strided window must alias the parent, not copy it.
    I8Tensor t(6, 6);
    const I8View v = t.view();
    for (uint32_t r = 0; r < 6; ++r) {
        for (uint32_t c = 0; c < 6; ++c) v.at(r, c) = static_cast<i8>(r * 10 + c);
    }

    const I8View sub = v.tile(1, 2, 2, 3);
    REQUIRE(sub.rows() == 2);
    REQUIRE(sub.cols() == 3);
    REQUIRE(sub.stride() == 6);          // the pitch is inherited from the parent
    for (uint32_t r = 0; r < 2; ++r) {
        for (uint32_t c = 0; c < 3; ++c) {
            REQUIRE(sub.at(r, c) == t.data()[(1 + r) * 6 + (2 + c)]);
        }
    }

    sub.at(1, 2) = 99;                   // written through the view
    REQUIRE(t.data()[2 * 6 + 4] == 99);

    // A window of a window composes.
    const I8View sub2 = sub.tile(0, 1, 1, 2);
    REQUIRE(sub2.at(0, 0) == v.at(1, 3));

    // A 2x3 source into a 4x4 tile zero-pads the remainder.
    I8Tensor src(2, 3);
    src.view().fill(7);
    I8Tensor dst(4, 4);
    dst.view().fill(-1);
    dst.view().zero_pad_from(src.view());
    std::vector<int> got, want;
    for (uint32_t r = 0; r < 4; ++r) {
        for (uint32_t c = 0; c < 4; ++c) {
            got.push_back(dst.view().at(r, c));
            want.push_back((r < 2 && c < 3) ? 7 : 0);
        }
    }
    REQUIRE_MSG(got == want, diff_vec("padded tile", got, want));

    // copy_from crosses strides.
    I8Tensor wide(4, 8);
    wide.view().fill(0);
    wide.view().tile(0, 4, 2, 3).copy_from(src.view());
    REQUIRE(wide.view().at(0, 4) == 7);
    REQUIRE(wide.view().at(1, 6) == 7);
    REQUIRE(wide.view().at(0, 7) == 0);

    // same_values reports a shape mismatch rather than asserting.
    REQUIRE(same_values(src.view(), src.view()));
    REQUIRE(!same_values(src.view(), dst.view()));

    // A non-const view converts to a const one.
    const ConstI8View cv = v;
    REQUIRE(cv.at(2, 4) == 99);
}

SECTION("quant") {
    using namespace quant;

    // Ties round away from zero, in both signs.
    REQUIRE(round_shift(5, 1) == 3);
    REQUIRE(round_shift(-5, 1) == -3);
    REQUIRE(round_shift(3, 1) == 2);
    REQUIRE(round_shift(-3, 1) == -2);
    REQUIRE(round_shift(1, 1) == 1);
    REQUIRE(round_shift(-1, 1) == -1);
    REQUIRE(round_shift(7, 0) == 7);

    // A bare arithmetic shift floors, which would give -1 and -2 here.
    REQUIRE(round_shift(-6, 2) == -2);
    REQUIRE(round_shift(-10, 2) == -3);

    // Average pooling's divisor is a window area, not a power of two.
    REQUIRE(round_div(5, 2) == 3);
    REQUIRE(round_div(-5, 2) == -3);
    REQUIRE(round_div(1, 3) == 0);
    REQUIRE(round_div(2, 3) == 1);
    REQUIRE(round_div(-2, 3) == -1);
    REQUIRE(round_div(14, 4) == 4);      // 3.5 -> 4
    REQUIRE(round_div(-14, 4) == -4);

    // Saturation clamps at both ends, not just the positive one.
    REQUIRE(requantize(127, 1, 0) == 127);
    REQUIRE(requantize(128, 1, 0) == 127);
    REQUIRE(requantize(100000, 1, 0) == 127);
    REQUIRE(requantize(-128, 1, 0) == -128);
    REQUIRE(requantize(-129, 1, 0) == -128);
    REQUIRE(requantize(-100000, 1, 0) == -128);

    // Bias folds in, and folding in a zero bias changes nothing.
    REQUIRE(requantize_biased(100, 0, 1, 0) == requantize(100, 1, 0));
    REQUIRE(requantize_biased(-5, 10, 1, 1) == 3);      // (-5+10)/2 = 2.5 -> 3
    REQUIRE(requantize_biased(200, 10, 1, 1) == 105);
    // A bias large enough to overflow int32 if it were added there.
    REQUIRE(requantize_biased(2147483647, 2147483647, 1, 8) == 127);

    // An independent rounding rule: compare twice the remainder to the divisor.
    auto expect_rq = [](i32 acc, i32 mult, uint32_t shift) -> int {
        const int64_t num = static_cast<int64_t>(acc) * mult;
        const int64_t den = int64_t{1} << shift;
        const int64_t a   = num < 0 ? -num : num;
        int64_t q = a / den;
        if (2 * (a % den) >= den) ++q;
        if (num < 0) q = -q;
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        return static_cast<int>(q);
    };

    int mismatches = 0;
    for (const i32 mult : std::initializer_list<i32>{1, 3, 127, 1000000}) {
        for (const uint32_t shift : std::initializer_list<uint32_t>{0u, 1u, 7u, 15u, 31u}) {
            for (i32 acc = -2000; acc <= 2000; ++acc) {
                if (requantize(acc, mult, shift) != expect_rq(acc, mult, shift)) ++mismatches;
            }
        }
    }
    REQUIRE(mismatches == 0);
}

