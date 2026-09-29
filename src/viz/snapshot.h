/// The visualizer's view of the machine: everything it draws, as one line of JSON.

#pragma once

#include <string>

#include "core/tpu.h"
#include "ui/shell.h"

// The machine's state and each unit's status after the last command; the values themselves come from MachineState.
std::string snapshot_json(const Tpu& tpu);
