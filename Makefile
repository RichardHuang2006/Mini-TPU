# Mini-TPU: cycle-accurate TPUv1-style int8 inference accelerator
#
# Targets: all (release), debug (ASan+UBSan), test, examples, report, clean, help

CXX      ?= g++
CXXSTD    = -std=c++17
CXXWARN   = -Wall -Wextra -Wpedantic
CXXINC    = -Isrc -Itests

CXXFLAGS_REL = $(CXXSTD) -O2 $(CXXWARN) $(CXXINC)
CXXFLAGS_DBG = $(CXXSTD) -O1 -g $(CXXWARN) $(CXXINC) -fsanitize=address,undefined
LDFLAGS_DBG  = -fsanitize=address,undefined

BUILD  = build
OBJDIR = $(BUILD)/obj
DBGDIR = $(BUILD)/obj-dbg

# Wildcards, so a new source file needs no Makefile edit.
TPU_SRC = $(wildcard src/*.cpp)
TPU_OBJ = $(patsubst src/%.cpp,$(OBJDIR)/%.o,$(TPU_SRC))
DBG_OBJ = $(patsubst src/%.cpp,$(DBGDIR)/%.o,$(TPU_SRC))

TEST_SRC   = $(wildcard tests/test_*.cpp)
TOOLS_SRC  = tools/gen_examples.cpp
REPORT_SRC = tools/report.cpp
HDR        = $(wildcard src/*.h) $(wildcard tests/*.h)

# The tests reuse every src/*.cpp but main.cpp, whose main() they replace by
# #including it.
LIB_SRC = $(filter-out src/main.cpp,$(TPU_SRC))

.PHONY: all debug test examples report clean help
.DEFAULT_GOAL := all

# ---------------------------------------------------------------- release ---
all: $(BUILD)/minitpu

$(BUILD)/minitpu: $(TPU_OBJ) | $(BUILD)
	@if [ -z "$(TPU_OBJ)" ]; then echo "no src/*.cpp to build"; exit 1; fi
	$(CXX) $(CXXFLAGS_REL) $(TPU_OBJ) -o $@

$(OBJDIR)/%.o: src/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS_REL) -MMD -MP -c $< -o $@

# ------------------------------------------------------ debug / sanitized ---
# The instrumented build exists to run the whole suite under it, so this target
# builds and runs rather than only building.
debug: $(BUILD)/minitpu-debug $(BUILD)/test_main-debug
	./$(BUILD)/test_main-debug

$(BUILD)/minitpu-debug: $(DBG_OBJ) | $(BUILD)
	@if [ -z "$(DBG_OBJ)" ]; then echo "no src/*.cpp to build"; exit 1; fi
	$(CXX) $(CXXFLAGS_DBG) $(DBG_OBJ) -o $@ $(LDFLAGS_DBG)

$(DBGDIR)/%.o: src/%.cpp | $(DBGDIR)
	$(CXX) $(CXXFLAGS_DBG) -MMD -MP -c $< -o $@

# --------------------------------------------------------------- tooling ---
$(BUILD)/gen_examples: $(TOOLS_SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS_REL) $(TOOLS_SRC) -o $@

# Skipped while the generator does not exist, rather than failing on a
# prerequisite that cannot be built.
examples:
	@if [ -f $(TOOLS_SRC) ]; then \
	  $(MAKE) --no-print-directory $(BUILD)/gen_examples && \
	  mkdir -p examples && ./$(BUILD)/gen_examples; \
	else \
	  echo "examples: skipped, no $(TOOLS_SRC)"; \
	fi

# Performance tables regenerated from the model, so the checked-in numbers are
# reproducible.
$(BUILD)/report: $(REPORT_SRC) $(LIB_SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS_REL) $(REPORT_SRC) $(LIB_SRC) -o $@

report: $(BUILD)/report
	@./$(BUILD)/report

# ----------------------------------------------------------------- trace ---
# The trace generator instruments a run through the TraceSink in src/trace.h
# and writes the .mtpt containers the visualizer under viz/ opens. It checks
# that tracing left the run unchanged and that the trace agrees with the
# oracle, the golden loops and itself, and fails if any check does.
TRACE_SRC = tools/tracegen.cpp
TRACE_DIR = viz/traces

$(BUILD)/tracegen: $(TRACE_SRC) $(LIB_SRC) $(HDR) | $(BUILD)
	$(CXX) $(CXXFLAGS_REL) $(TRACE_SRC) $(LIB_SRC) -o $@

trace: $(BUILD)/tracegen examples
	@mkdir -p $(TRACE_DIR)
	./$(BUILD)/tracegen --small --out $(TRACE_DIR)/matmul_8.mtpt
	./$(BUILD)/tracegen --prog examples/matmul_128.hex \
	    --acts examples/matmul_128.acts.mtpu --weights examples/matmul_128.weights.mtpu \
	    --expect examples/matmul_128.expect.mtpu --layer 128,128,128 \
	    --dim 32 --ub 262144 --acc-banks 4 --macs 2097152 \
	    --out $(TRACE_DIR)/matmul_128.mtpt
	python3 viz/check_trace.py $(TRACE_DIR)/matmul_8.mtpt $(TRACE_DIR)/matmul_128.mtpt
	python3 viz/embed_small.py $(TRACE_DIR)/matmul_8.mtpt viz/traces/matmul_8.js

# ------------------------------------------------------------------ test ---
# One translation unit per subsystem (tests/test_*.cpp), linked into a single
# binary whose main() lives in tests/test_main.cpp. Every TU pulls in nearly
# every header, so the suite is rebuilt on any header change rather than
# tracked dependency by dependency.
$(BUILD)/test_main: $(TEST_SRC) $(TPU_SRC) $(HDR) | $(BUILD)
	@if [ -z "$(TEST_SRC)" ]; then echo "no tests/test_*.cpp"; exit 1; fi
	$(CXX) $(CXXFLAGS_REL) $(TEST_SRC) $(LIB_SRC) -o $@

$(BUILD)/test_main-debug: $(TEST_SRC) $(TPU_SRC) $(HDR) | $(BUILD)
	@if [ -z "$(TEST_SRC)" ]; then echo "no tests/test_*.cpp"; exit 1; fi
	$(CXX) $(CXXFLAGS_DBG) $(TEST_SRC) $(LIB_SRC) -o $@ $(LDFLAGS_DBG)

test: $(BUILD)/test_main examples
	./$(BUILD)/test_main

# ---------------------------------------------------------------- housekeeping
$(BUILD) $(OBJDIR) $(DBGDIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD) examples

help:
	@echo "Mini-TPU targets:"
	@echo "  all       build/minitpu         (release, -O2, warnings on)"
	@echo "  debug     build/minitpu-debug   (ASan + UBSan, -O1 -g) + run the suite sanitized"
	@echo "  test      compile+run the test suite after regenerating examples/"
	@echo "  examples  write the bundled workloads to examples/"
	@echo "  report    regenerate the performance tables from the model"
	@echo "  clean     remove build/ and examples/"

# Auto-generated header dependencies.
-include $(TPU_OBJ:.o=.d) $(DBG_OBJ:.o=.d)
