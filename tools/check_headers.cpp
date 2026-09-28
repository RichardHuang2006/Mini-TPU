/// Lint rule: a source file opens with one `/// ...` line, and no other comment comes before its first line of code.

#include <cstdio>
#include <fstream>
#include <string>

namespace {

struct Finding {
    int         line = 0;   // 0 means the file is clean
    std::string message;
};

bool is_blank(const std::string& line) {
    return line.find_first_not_of(" \t\r") == std::string::npos;
}

bool starts_with(const std::string& text, const std::string& prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

bool is_doc_line(const std::string& line) {
    if (!starts_with(line, "/// ")) {
        return false;
    }
    const std::string description = line.substr(4);
    return !is_blank(description);
}

bool is_comment(const std::string& line) {
    const std::size_t first_char = line.find_first_not_of(" \t");
    const std::string trimmed = line.substr(first_char);
    return starts_with(trimmed, "//") || starts_with(trimmed, "/*");
}

Finding check(const char* path) {
    std::ifstream file(path);
    if (!file) {
        return {1, "cannot open file"};
    }

    std::string line;
    const bool has_first_line = static_cast<bool>(std::getline(file, line));
    if (!has_first_line || !is_doc_line(line)) {
        return {1, "the first line must be a `/// ...` doc line"};
    }

    // Everything between the doc line and the first line of code must be blank.
    int number = 1;
    while (std::getline(file, line)) {
        number = number + 1;
        if (is_blank(line)) {
            continue;
        }
        if (is_comment(line)) {
            return {number, "only the `///` line may be a comment above the code"};
        }
        break;
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    int bad_files = 0;
    for (int i = 1; i < argc; ++i) {
        const Finding finding = check(argv[i]);
        if (finding.line == 0) {
            continue;
        }
        std::printf("%s:%d: error: %s\n", argv[i], finding.line, finding.message.c_str());
        bad_files = bad_files + 1;
    }

    if (bad_files > 0) {
        return 1;
    }
    return 0;
}
