// Tests for the processing element and matrix unit of src/systolic_array.h.

#include "test_support.h"

namespace {

// The same matmul on the cycle-accurate array.
std::vector<int> mxu_matmul(const Config& cfg, const std::vector<i8>& weights,
                            const std::vector<i8>& acts, uint32_t len) {
    Mxu mxu(cfg);
    mxu.load_weights(ConstI8View(weights.data(), cfg.dim, cfg.dim));

    I32Tensor out(len == 0 ? 1 : len, cfg.dim);
    if (len != 0) {
        mxu.matmul(ConstI8View(acts.data(), len, cfg.dim), out.view(), false);
    }

    std::vector<int> flat;
    flat.reserve(static_cast<std::size_t>(len) * cfg.dim);
    for (uint32_t r = 0; r < len; ++r) {
        for (uint32_t c = 0; c < cfg.dim; ++c) flat.push_back(out.view().at(r, c));
    }
    return flat;
}

}  // namespace

SECTION("pe") {
    Pe p;
    p.set_weight(3, 0);
    p.set_weight(-5, 1);
    REQUIRE(p.weight(0) == 3);
    REQUIRE(p.weight(1) == -5);

    // Nothing is visible until commit, so PE visit order cannot leak out.
    p.tick(7, 100, 0);
    REQUIRE(p.act_out() == 0);
    REQUIRE(p.psum_out() == 0);

    p.commit();
    REQUIRE(p.psum_out() == 100 + 7 * 3);
    REQUIRE(p.act_out() == 7);       // the activation passed through as well

    // The plane argument selects which weight multiplies.
    p.tick(7, 0, 1);
    p.commit();
    REQUIRE(p.psum_out() == 7 * -5);

    // A held value is not sticky: feeding zero clears the products back out.
    p.tick(0, 0, 0);
    p.commit();
    REQUIRE(p.psum_out() == 0);
    REQUIRE(p.act_out() == 0);

    // The extremes of int8, including the asymmetric -128.
    Pe q;
    q.set_weight(-128, 0);
    q.tick(-128, 0, 0);
    q.commit();
    REQUIRE(q.psum_out() == 16384);

    q.set_weight(127, 0);
    q.tick(-128, 0, 0);
    q.commit();
    REQUIRE(q.psum_out() == -16256);

    // clear_pipeline drops data in flight but keeps the weights.
    q.tick(5, 999, 0);
    q.commit();
    REQUIRE(q.psum_out() != 0);
    q.clear_pipeline();
    REQUIRE(q.psum_out() == 0);
    REQUIRE(q.act_out() == 0);
    REQUIRE(q.weight(0) == 127);

    // A chain of PEs delays by exactly one cycle per hop.
    Pe chain[3];
    for (Pe& c : chain) c.set_weight(1, 0);
    for (int cycle = 0; cycle < 3; ++cycle) {
        const i8 in = (cycle == 0) ? i8{42} : i8{0};
        chain[0].tick(in, 0, 0);
        chain[1].tick(chain[0].act_out(), 0, 0);
        chain[2].tick(chain[1].act_out(), 0, 0);
        for (Pe& c : chain) c.commit();
        REQUIRE(chain[2].act_out() == (cycle == 2 ? 42 : 0));
    }
}

SECTION("mxu_weights") {
    Config cfg = small_cfg();        // dim 4
    cfg.double_buffer = false;       // load straight into the active plane

    Mxu mxu(cfg);
    REQUIRE(mxu.dim() == 4);
    REQUIRE(mxu.active_plane() == 0);
    REQUIRE(mxu.load_plane() == 0);

    // PE[k][c] must hold W[k][c] and not its transpose.
    I8Tensor w(4, 4);
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) w.view().at(k, c) = static_cast<i8>(k * 10 + c);
    }
    mxu.load_weights(w.view());

    bool placed = true;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) {
            if (mxu.weight_at(k, c, 0) != static_cast<i8>(k * 10 + c)) placed = false;
        }
    }
    REQUIRE(placed);
    REQUIRE(mxu.weight_at(0, 3, 0) == 3);     // and not 30
    REQUIRE(mxu.weight_at(3, 0, 0) == 30);
    REQUIRE(mxu.plane_rows(0) == 4);
    REQUIRE(mxu.plane_cols(0) == 4);
    REQUIRE(mxu.stats().weight_loads == 1);

    // A partial tile zero-pads the rest of the array.
    Mxu part(cfg);
    I8Tensor small(2, 3);
    small.view().fill(9);
    part.load_weights(small.view());

    std::vector<int> got, want;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 4; ++c) {
            got.push_back(part.weight_at(k, c, 0));
            want.push_back((k < 2 && c < 3) ? 9 : 0);
        }
    }
    REQUIRE_MSG(got == want, diff_vec("padded weights", got, want));
    REQUIRE(part.plane_rows(0) == 2);
    REQUIRE(part.plane_cols(0) == 3);

    // Loading again must overwrite the padding a larger tile left behind.
    part.load_weights(w.view());
    REQUIRE(part.weight_at(3, 3, 0) == 33);
    I8Tensor tiny(1, 1);
    tiny.view().fill(4);
    part.load_weights(tiny.view());
    REQUIRE(part.weight_at(0, 0, 0) == 4);
    REQUIRE(part.weight_at(3, 3, 0) == 0);
    REQUIRE(part.weight_at(0, 1, 0) == 0);

    // With double buffering the load leaves the active plane untouched.
    Config db = small_cfg();
    db.double_buffer = true;
    Mxu two(db);
    REQUIRE(two.load_plane() == 1);
    two.load_weights(w.view());
    REQUIRE(two.weight_at(2, 2, 1) == 22);
    REQUIRE(two.weight_at(2, 2, 0) == 0);      // active plane still clear
    REQUIRE(two.switch_pending());
}

SECTION("mxu_matmul") {
    // The array against the oracle, over shapes covering both int8 extremes.
    Lcg rng(12345);
    int shapes = 0, disagreements = 0;
    std::string first_bad;

    for (const uint32_t dim : std::initializer_list<uint32_t>{1u, 2u, 3u, 4u, 8u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = false;

        for (const uint32_t len : std::initializer_list<uint32_t>{1u, 2u, 3u, dim, dim}) {
            if (len > dim) continue;             // a bank holds at most dim rows

            std::vector<i8> w(static_cast<std::size_t>(dim) * dim);
            std::vector<i8> a(static_cast<std::size_t>(len) * dim);
            for (i8& v : w) v = rng.byte();
            for (i8& v : a) v = rng.byte();

            ++shapes;
            const std::vector<int> want = ref_matmul(cfg, w, a, len);
            const std::vector<int> got  = mxu_matmul(cfg, w, a, len);
            if (got != want) {
                ++disagreements;
                if (first_bad.empty()) {
                    first_bad = "    dim " + std::to_string(dim) + " len " +
                                std::to_string(len) + "\n" + diff_vec("out", got, want);
                }
            }
        }
    }
    REQUIRE(shapes > 0);
    REQUIRE_MSG(disagreements == 0, first_bad);

    // Every product is 16384, the widest the int32 chain gets in these tests.
    {
        Config cfg = small_cfg(8, 2);
        cfg.double_buffer = false;
        const std::vector<i8> w(64, -128);
        const std::vector<i8> a(64, -128);
        const std::vector<int> want = ref_matmul(cfg, w, a, 8);
        const std::vector<int> got  = mxu_matmul(cfg, w, a, 8);
        REQUIRE_MSG(got == want, diff_vec("extremes", got, want));
        REQUIRE(want[0] == 8 * 16384);
    }

    // Row [1 0 0 0] selects W row 0; [0 1 1 0] sums W rows 1 and 2.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);
        I8Tensor w(4, 4);
        for (uint32_t k = 0; k < 4; ++k) {
            for (uint32_t c = 0; c < 4; ++c) {
                w.view().at(k, c) = static_cast<i8>(k * 4 + c + 1);
            }
        }
        mxu.load_weights(w.view());

        I8Tensor a(2, 4);
        a.view().fill(0);
        a.view().at(0, 0) = 1;
        a.view().at(1, 1) = 1;
        a.view().at(1, 2) = 1;

        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want{1, 2, 3, 4, 14, 16, 18, 20};
        REQUIRE_MSG(got == want, diff_vec("hand-checked", got, want));
    }

    // A real 2x3 by 3x2 problem, zero padded into a 4x4 array.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor w(3, 2);
        w.view().at(0, 0) = 1; w.view().at(0, 1) = 2;
        w.view().at(1, 0) = 3; w.view().at(1, 1) = 4;
        w.view().at(2, 0) = 5; w.view().at(2, 1) = 6;
        mxu.load_weights(w.view());

        I8Tensor a(2, 4);
        a.view().fill(0);
        a.view().at(0, 0) = 1; a.view().at(0, 2) = 1;
        a.view().at(1, 0) = 2; a.view().at(1, 1) = 1;

        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 2; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want{6, 8, 0, 0, 5, 8, 0, 0};
        REQUIRE_MSG(got == want, diff_vec("partial tile", got, want));
    }

    // Accumulating into a bank that already holds a previous K-tile's result.
    {
        Config cfg = small_cfg();
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor ident(4, 4);
        ident.view().fill(0);
        for (uint32_t i = 0; i < 4; ++i) ident.view().at(i, i) = 1;
        I8Tensor twos(4, 4);
        twos.view().fill(2);

        I8Tensor a0(1, 4), a1(1, 4);
        for (uint32_t k = 0; k < 4; ++k) a0.view().at(0, k) = static_cast<i8>(k + 1);
        a1.view().fill(1);

        I32Tensor out(1, 4);
        mxu.load_weights(ident.view());
        mxu.matmul(a0.view(), out.view(), false);
        mxu.load_weights(twos.view());
        mxu.matmul(a1.view(), out.view(), true);

        std::vector<int> got;
        for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(0, c));
        const std::vector<int> want{9, 10, 11, 12};
        REQUIRE_MSG(got == want, diff_vec("k-tiled", got, want));
    }

    // Back-to-back matmuls must not leak: the de-skew window has to reject the
    // previous stream still draining. Large values first, so a leak is obvious.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = false;

        const std::vector<i8> big_w(16, 127);
        const std::vector<i8> big_a(16, 127);
        std::vector<i8>       w2(16), a2(16);
        Lcg r2(999);
        for (i8& v : w2) v = r2.byte();
        for (i8& v : a2) v = r2.byte();

        Mxu mxu(cfg);
        I32Tensor scratch(4, 4), out(4, 4);
        mxu.load_weights(ConstI8View(big_w.data(), 4, 4));
        mxu.matmul(ConstI8View(big_a.data(), 4, 4), scratch.view(), false);
        mxu.load_weights(ConstI8View(w2.data(), 4, 4));
        mxu.matmul(ConstI8View(a2.data(), 4, 4), out.view(), false);

        std::vector<int> got;
        for (uint32_t r = 0; r < 4; ++r) {
            for (uint32_t c = 0; c < 4; ++c) got.push_back(out.view().at(r, c));
        }
        const std::vector<int> want = ref_matmul(cfg, w2, a2, 4);
        REQUIRE_MSG(got == want, diff_vec("second matmul", got, want));
    }
}

SECTION("mxu_timing") {
    // The occupancy formula, over several shapes.
    for (const uint32_t dim : std::initializer_list<uint32_t>{1u, 2u, 4u, 8u}) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = false;

        for (uint32_t len = 1; len <= dim; ++len) {
            Mxu mxu(cfg);
            I8Tensor w(dim, dim);
            w.view().fill(1);
            mxu.load_weights(w.view());

            I8Tensor a(len, dim);
            a.view().fill(1);
            I32Tensor out(len, dim);

            const uint64_t before = mxu.cycle();
            const MxuTiming t = mxu.matmul(a.view(), out.view(), false);

            REQUIRE(t.cycles == static_cast<uint64_t>(len) + 2u * dim - 1u);
            REQUIRE(t.start == before);
            REQUIRE(t.end == t.start + t.cycles);
            REQUIRE(mxu.cycle() == t.end);

            // A row completes r + 2*dim - 1 cycles after the stream began.
            REQUIRE(t.row_valid.size() == len);
            bool valid_ok = true;
            for (uint32_t r = 0; r < len; ++r) {
                if (t.row_valid[r] != t.start + r + 2ull * dim - 1ull) valid_ok = false;
                if (r > 0 && t.row_valid[r] <= t.row_valid[r - 1]) valid_ok = false;
            }
            REQUIRE(valid_ok);

            // The last row lands on the final cycle of the occupancy window.
            REQUIRE(t.row_valid.back() == t.end - 1);
            REQUIRE(t.row_valid.front() >= t.start);
        }
    }

    // Utilization is len / (len + 2*dim - 1), with double buffering on so that
    // fill and drain is the only thing measured.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = true;

        Mxu one(cfg);
        I8Tensor w(4, 4);
        w.view().fill(1);
        I8Tensor a1(1, 4);
        a1.view().fill(1);
        I32Tensor o1(1, 4);
        one.load_weights(w.view());
        one.matmul(a1.view(), o1.view(), false);
        REQUIRE(one.stats().cycles == 1 + 7);
        REQUIRE(one.stats().fill_drain_cycles == 7);
        REQUIRE(one.stats().useful_macs == 16);
        REQUIRE(one.utilization() < 0.2);        // 16 / (16 * 8)

        // Four times the work in less than twice the time.
        Mxu four(cfg);
        I8Tensor a4(4, 4);
        a4.view().fill(1);
        I32Tensor o4(4, 4);
        four.load_weights(w.view());
        four.matmul(a4.view(), o4.view(), false);
        REQUIRE(four.stats().cycles == 4 + 7);
        REQUIRE(four.utilization() > one.utilization());

        // A stalled load shows up as lost utilization, not as missing time.
        Config slow = cfg;
        slow.double_buffer = false;
        Mxu stalled(slow);
        I32Tensor os(4, 4);
        stalled.load_weights(w.view());
        stalled.matmul(a4.view(), os.view(), false);
        REQUIRE(stalled.stats().cycles == 4 + 7 + 4);
        REQUIRE(stalled.stats().useful_macs == four.stats().useful_macs);
        REQUIRE(stalled.utilization() < four.utilization());
    }

    // A real 2x3 tile in a 4x4 array: 2*2*3 = 12 useful MACs of 2*16 = 32 slots.
    {
        Config cfg = small_cfg(4, 2);
        cfg.double_buffer = false;
        Mxu mxu(cfg);

        I8Tensor small(2, 3);
        small.view().fill(1);
        mxu.load_weights(small.view());

        I8Tensor a(2, 4);
        a.view().fill(1);
        I32Tensor out(2, 4);
        mxu.matmul(a.view(), out.view(), false);

        REQUIRE(mxu.stats().useful_macs == 12);
        REQUIRE(mxu.stats().partial_tile_waste == 20);

        // A full tile wastes nothing.
        Mxu full(cfg);
        I8Tensor w(4, 4);
        w.view().fill(1);
        full.load_weights(w.view());
        I32Tensor fout(2, 4);
        full.matmul(a.view(), fout.view(), false);
        REQUIRE(full.stats().useful_macs == 32);
        REQUIRE(full.stats().partial_tile_waste == 0);
    }
}

SECTION("mxu_double_buffer") {
    // The second tile's load is free into the shadow plane, dim cycles without.
    const uint32_t dim = 4;

    I8Tensor w0(dim, dim), w1(dim, dim);
    for (uint32_t k = 0; k < dim; ++k) {
        for (uint32_t c = 0; c < dim; ++c) {
            w0.view().at(k, c) = static_cast<i8>(k + c + 1);
            w1.view().at(k, c) = static_cast<i8>(k * 2 + 1);
        }
    }
    I8Tensor a(dim, dim);
    for (uint32_t r = 0; r < dim; ++r) {
        for (uint32_t c = 0; c < dim; ++c) a.view().at(r, c) = static_cast<i8>(r + c);
    }

    // Reports the bubble charged to the second weight load, plus both results.
    auto run_pair = [&](bool double_buffer, uint64_t& second_load_bubble,
                        std::vector<int>& r0, std::vector<int>& r1) {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = double_buffer;

        Mxu mxu(cfg);
        I32Tensor o0(dim, dim), o1(dim, dim);

        mxu.load_weights(w0.view());
        mxu.matmul(a.view(), o0.view(), false);

        const uint64_t before = mxu.cycle();
        mxu.load_weights(w1.view());
        second_load_bubble = mxu.cycle() - before;

        mxu.matmul(a.view(), o1.view(), false);

        r0.clear();
        r1.clear();
        for (uint32_t r = 0; r < dim; ++r) {
            for (uint32_t c = 0; c < dim; ++c) {
                r0.push_back(o0.view().at(r, c));
                r1.push_back(o1.view().at(r, c));
            }
        }
        return mxu.stats();
    };

    uint64_t db_bubble = 0, nodb_bubble = 0;
    std::vector<int> db_r0, db_r1, nodb_r0, nodb_r1;
    const MxuStats db   = run_pair(true, db_bubble, db_r0, db_r1);
    const MxuStats nodb = run_pair(false, nodb_bubble, nodb_r0, nodb_r1);

    REQUIRE(db_bubble == 0);
    REQUIRE(nodb_bubble == dim);
    REQUIRE(db.weight_load_bubble == 0);
    REQUIRE(nodb.weight_load_bubble == 2ull * dim);   // both loads are exposed

    // The whole difference between the two configurations is the weight bubbles.
    const uint64_t compute = 2ull * (dim + 2ull * dim - 1ull);
    REQUIRE(db.cycles == compute);
    REQUIRE(nodb.cycles == compute + 2ull * dim);

    // Speed must not change the answer; a skipped load would also report no bubble.
    REQUIRE_MSG(db_r0 == nodb_r0, diff_vec("first matmul", db_r0, nodb_r0));
    REQUIRE_MSG(db_r1 == nodb_r1, diff_vec("second matmul", db_r1, nodb_r1));

    // Both agree with the oracle, so the second tile really did become resident.
    {
        Config cfg = small_cfg(dim, 2);
        std::vector<i8> fw0, fw1, fa;
        for (uint32_t k = 0; k < dim; ++k) {
            for (uint32_t c = 0; c < dim; ++c) {
                fw0.push_back(w0.view().at(k, c));
                fw1.push_back(w1.view().at(k, c));
                fa.push_back(a.view().at(k, c));
            }
        }
        const std::vector<int> want0 = ref_matmul(cfg, fw0, fa, dim);
        const std::vector<int> want1 = ref_matmul(cfg, fw1, fa, dim);
        REQUIRE_MSG(db_r0 == want0, diff_vec("tile 0 vs oracle", db_r0, want0));
        REQUIRE_MSG(db_r1 == want1, diff_vec("tile 1 vs oracle", db_r1, want1));
        REQUIRE(want0 != want1);      // the two tiles really do differ
    }

    // Double buffered, the active plane alternates; otherwise it stays at 0.
    REQUIRE(db.plane_switches == 2);
    REQUIRE(nodb.plane_switches == 0);

    {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = true;
        Mxu mxu(cfg);
        I32Tensor out(dim, dim);

        REQUIRE(mxu.active_plane() == 0);
        REQUIRE(mxu.load_plane() == 1);
        mxu.load_weights(w0.view());
        REQUIRE(mxu.switch_pending());
        REQUIRE(mxu.active_plane() == 0);        // not yet; the matmul switches

        mxu.matmul(a.view(), out.view(), false);
        REQUIRE(!mxu.switch_pending());
        REQUIRE(mxu.active_plane() == 1);
        REQUIRE(mxu.load_plane() == 0);          // the next load goes to the other

        mxu.load_weights(w1.view());
        mxu.matmul(a.view(), out.view(), false);
        REQUIRE(mxu.active_plane() == 0);

        // The retired tile is still sitting in the shadow plane, untouched.
        REQUIRE(mxu.weight_at(0, 0, 1) == w0.view().at(0, 0));
    }

    // A plane switch is charged to a load, not to every matmul.
    {
        Config cfg = small_cfg(dim, 2);
        cfg.double_buffer = true;
        Mxu mxu(cfg);
        I32Tensor o0(dim, dim), o1(dim, dim);
        mxu.load_weights(w0.view());
        mxu.matmul(a.view(), o0.view(), false);
        mxu.matmul(a.view(), o1.view(), false);
        REQUIRE(mxu.stats().plane_switches == 1);
        REQUIRE(mxu.active_plane() == 1);

        std::vector<int> f0, f1;
        for (uint32_t r = 0; r < dim; ++r) {
            for (uint32_t c = 0; c < dim; ++c) {
                f0.push_back(o0.view().at(r, c));
                f1.push_back(o1.view().at(r, c));
            }
        }
        REQUIRE_MSG(f0 == f1, diff_vec("repeated matmul", f0, f1));
    }
}

