/// Lint rule: a source file opens with one `/// ...` line, and no other comment comes before its first line of code.

#include <cstdio>
#include <fstream>
#include <string>

namespace {

struct Finding {
    int         line = 0;   // 0: the file is clean
    std::string what;
};

bool blank(const std::string& s) { return s.find_first_not_of(" \t\r") == std::string::npos; }

Finding check(const char* path) {
    std::ifstream in(path);
    if (!in) return {1, "cannot open file"};

    std::string line;
    if (!std::getline(in, line) || line.rfind("/// ", 0) != 0 || blank(line.substr(4)))
        return {1, "the first line must be a `/// ...` doc line"};

    for (int n = 2; std::getline(in, line); ++n) {
        if (blank(line)) continue;
        const std::string code = line.substr(line.find_first_not_of(" \t"));
        if (code.rfind("//", 0) == 0 || code.rfind("/*", 0) == 0)
            return {n, "only the `///` line may be a comment above the code"};
        break;
    }
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    int bad = 0;
    for (int i = 1; i < argc; ++i) {
        const Finding f = check(argv[i]);
        if (f.line == 0) continue;
        std::printf("%s:%d: error: %s\n", argv[i], f.line, f.what.c_str());
        ++bad;
    }
    return bad == 0 ? 0 : 1;
}
