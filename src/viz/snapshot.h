/// The visualizer's view of the machine: everything it draws, as one line of JSON.

#pragma once

#include <string>

#include "core/tpu.h"
#include "ui/shell.h"

// The whole visible state after the last command: every unit, the memory maps, the timeline and the pinned views.
std::string snapshot_json(const Tpu& tpu, const Shell& shell);
