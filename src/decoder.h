#pragma once

#include "isa.h"

// Wire form to field form. The only failure mode is an opcode outside the
// defined set, which decodes to a trapping HALT: a malformed program stops the
// machine rather than running on undefined state.
Decoded decode(const RawInst& raw);
