#pragma once

#include "isa.h"

// Wire form to field form. The only failure mode is an opcode outside the
// defined set, which decodes to a trapping HALT rather than to undefined
// behaviour -- a malformed program stops the machine, it does not wander.
Decoded decode(const RawInst& raw);
