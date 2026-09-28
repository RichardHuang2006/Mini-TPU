/// The command loop: an interactive `tpu>` prompt with history, or plain line mode when input is piped.

#pragma once

#include <iosfwd>

#include "ui/shell.h"

// Reads commands at the `tpu>` prompt until quit, Ctrl-C or Ctrl-D, printing each command's output.
void run_interactive(Shell& shell);

// Piped input: echoes each command after `tpu> `, then prints its output.
void run_plain(Shell& shell, std::istream& in, std::ostream& out);
