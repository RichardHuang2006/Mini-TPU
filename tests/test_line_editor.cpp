/// Line editor: typing, backspace, and walking history with up and down.

#include <string>

#include "test_framework.h"
#include "ui/line_editor.h"

TEST(editor_types_and_takes) {
    LineEditor editor;
    editor.type('r');
    editor.type('u');
    editor.type('x');
    editor.backspace();
    editor.type('n');
    CHECK_EQ(editor.text(), std::string("run"));

    CHECK_EQ(editor.take(), std::string("run"));
    CHECK_EQ(editor.text(), std::string(""));

    editor.backspace();   // nothing to delete is fine
    CHECK_EQ(editor.text(), std::string(""));
}

TEST(editor_history_walks_both_ways) {
    LineEditor editor;
    editor.remember("load programs/copy.s");
    editor.remember("step 5");

    editor.older();
    CHECK_EQ(editor.text(), std::string("step 5"));
    editor.older();
    CHECK_EQ(editor.text(), std::string("load programs/copy.s"));
    editor.older();   // already at the oldest
    CHECK_EQ(editor.text(), std::string("load programs/copy.s"));

    editor.newer();
    CHECK_EQ(editor.text(), std::string("step 5"));
    editor.newer();   // past the newest is a fresh, empty line
    CHECK_EQ(editor.text(), std::string(""));
}

TEST(editor_skips_repeated_commands) {
    LineEditor editor;
    editor.remember("step");
    editor.remember("step");
    editor.remember("back");

    editor.older();
    editor.older();
    editor.older();
    CHECK_EQ(editor.text(), std::string("step"));   // two entries in history, not three
}
