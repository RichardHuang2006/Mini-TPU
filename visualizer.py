#!/usr/bin/env python3
"""Serve viz/ over local HTTP and open the Mini-TPU cycle visualizer."""
import argparse
import http.server
import json
import os
import socket
import subprocess
import sys
import threading
import webbrowser

EXAMPLES = """examples:
  python3 visualizer.py                        # matmul_128 if traced, else the embedded example
  python3 visualizer.py --trace matmul_8       # or a full path/.mtpt
  python3 visualizer.py --build                # regenerate traces first
  python3 visualizer.py --port 0 --no-browser  # print the URL only
"""

REPO = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(REPO, "viz")
TRACES = os.path.join(VIZ, "traces")
DEFAULT_TRACE = "matmul_128"


def trace_index():
    """Every container in viz/traces, for the page's Workload dropdown."""
    out = []
    if os.path.isdir(TRACES):
        for f in sorted(os.listdir(TRACES)):
            if f.endswith(".mtpt"):
                out.append({"name": f[:-5], "file": "traces/" + f,
                            "bytes": os.path.getsize(os.path.join(TRACES, f))})
    return out


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from viz/, plus a dynamic /traces/index.json."""

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=VIZ, **kw)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/traces/index.json":
            body = json.dumps(trace_index()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        # Keep the console to one line per container fetched.
        if ".mtpt" in fmt % args:
            super().log_message(fmt, *args)


def resolve_trace(name):
    """--trace accepts a bare workload name, a filename, or a path."""
    if not name:
        default = os.path.join(TRACES, DEFAULT_TRACE + ".mtpt")
        if os.path.isfile(default):
            return os.path.relpath(default, VIZ)
        return None
    candidates = [
        name,
        os.path.join(TRACES, name),
        os.path.join(TRACES, name + ".mtpt"),
        os.path.join(REPO, name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.relpath(c, VIZ)
    sys.exit(
        "visualizer.py: can't find a trace named %r (looked in %s)\n"
        "Run with --build, or `make trace`, to generate viz/traces/*.mtpt first."
        % (name, TRACES)
    )


def build_traces():
    print("visualizer.py: building trace containers (make trace)...")
    sdk = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
    env = dict(os.environ)
    make_cmd = ["make", "trace"]
    if sys.platform == "darwin" and os.path.isdir(sdk):
        # The system clang++ may be blocked by an unaccepted Xcode license.
        clt_clang = "/Library/Developer/CommandLineTools/usr/bin/clang++"
        if os.path.isfile(clt_clang):
            env["SDKROOT"] = sdk
            make_cmd = ["make", "CXX=" + clt_clang, "trace"]
    subprocess.run(make_cmd, cwd=REPO, env=env, check=True)


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog=EXAMPLES,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", metavar="NAME", help="workload name or path of a .mtpt container to open immediately (default: %s when traced)" % DEFAULT_TRACE)
    p.add_argument("--build", action="store_true", help="run `make trace` first to (re)generate the containers")
    p.add_argument("--port", type=int, default=8765, help="local port to serve on (0 = pick a free one; default 8765)")
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    args = p.parse_args()

    if args.build:
        build_traces()

    trace_rel = resolve_trace(args.trace)

    port = args.port or free_port()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    port = httpd.server_address[1]

    url = "http://127.0.0.1:%d/index.html" % port
    if trace_rel:
        url += "?trace=" + trace_rel

    print("visualizer.py: serving %s at %s" % (VIZ, url))
    names = [t["name"] for t in trace_index()]
    print("visualizer.py: opening %s; containers in viz/traces: %s"
          % (trace_rel or "the embedded matmul_8", ", ".join(names) or "none"))
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nvisualizer.py: stopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
