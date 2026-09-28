/// Weight FIFO: 1366 cycles per tile, tiles arriving in order, four slots, pop, and bad tiles.

#include <vector>

#include "test_framework.h"
#include "units/weight_fifo.h"

namespace {

// Puts `first` at the start of a tile and `last` at its end.
void mark_tile(Dram& wmem, u32 tile, i8 first, i8 last) {
    const u64 start = static_cast<u64>(tile) * v1::kTileBytes;
    wmem.write(start, &first, 1);
    wmem.write(start + v1::kTileBytes - 1, &last, 1);
}

void tick_times(WeightFifo& fifo, int cycles) {
    for (int i = 0; i < cycles; ++i) {
        fifo.tick();
    }
}

}  // namespace

TEST(wfifo_one_tile_takes_1366_cycles) {
    Dram wmem("wmem", v1::kWeightMemBytes);
    mark_tile(wmem, 2, 11, -11);
    WeightFifo fifo(wmem);

    fifo.push(2);
    CHECK(fifo.fetching());

    tick_times(fifo, 1365);   // 1365 cycles of 48 bytes is 65520; 16 bytes are left
    CHECK(!fifo.front_ready());
    CHECK_EQ(fifo.tiles().front().bytes_arrived, u64{65520});

    fifo.tick();
    CHECK(fifo.front_ready());
    CHECK(!fifo.fetching());

    const std::vector<i8> tile = fifo.pop();
    CHECK_EQ(tile.front(), 11);
    CHECK_EQ(tile.back(), -11);
    CHECK(fifo.tiles().empty());
}

TEST(wfifo_tiles_arrive_in_order) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    WeightFifo fifo(wmem);
    fifo.push(0);
    fifo.push(1);

    tick_times(fifo, 1366);
    CHECK(fifo.tiles()[0].ready());
    CHECK_EQ(fifo.tiles()[1].bytes_arrived, u64{0});   // one DDR3 channel: the second waits its turn

    tick_times(fifo, 1366);
    CHECK(fifo.tiles()[1].ready());
}

TEST(wfifo_holds_four_tiles) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    WeightFifo fifo(wmem);
    for (u32 tile = 0; tile < 4; ++tile) {
        fifo.push(tile);
    }
    CHECK(fifo.full());
    CHECK_THROWS(fifo.push(4));
    CHECK_EQ(fifo.tiles().size(), std::size_t{4});
}

TEST(wfifo_pop_needs_a_ready_tile) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    WeightFifo fifo(wmem);
    CHECK_THROWS(fifo.pop());   // empty

    fifo.push(0);
    fifo.tick();
    CHECK_THROWS(fifo.pop());   // still arriving
}

TEST(wfifo_bad_tile_and_reset) {
    const Dram wmem("wmem", v1::kWeightMemBytes);
    WeightFifo fifo(wmem);
    const u32 tiles_in_8_gib = 131072;
    CHECK_THROWS(fifo.push(tiles_in_8_gib));
    fifo.push(tiles_in_8_gib - 1);   // the last tile is fine

    fifo.reset();
    CHECK(fifo.tiles().empty());
    CHECK(!fifo.fetching());
}
