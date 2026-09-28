/// The terminal front end: panels over a `tpu>` prompt when interactive, plain line mode when input is piped.

#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "core/tpu.h"
#include "ui/shell.h"

// The whole screen as `height` lines: panels exactly `width` columns wide, then the prompt line.
std::vector<std::string> render_screen(const Tpu& tpu, const std::string& file, const std::string& output,
                                       const std::string& input, int width, int height);

// Raw-mode terminal: redraws on every key until quit, Ctrl-C or Ctrl-D.
void run_interactive(Shell& shell, const Tpu& tpu, const std::string& greeting);

// Piped input: echoes each command after `tpu> `, then prints its output.
void run_plain(Shell& shell, std::istream& in, std::ostream& out);
