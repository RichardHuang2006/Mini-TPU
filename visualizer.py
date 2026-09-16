#!/usr/bin/env python3
"""Serve viz/ over local HTTP and open the Mini-TPU cycle visualizer."""
import argparse
import http.server
import os
import socket
import subprocess
import sys
import threading
import webbrowser

EXAMPLES = """examples:
  python3 visualizer.py                        # embedded small example
  python3 visualizer.py --trace matmul_128     # or a full path/.mtpt
  python3 visualizer.py --build                # regenerate traces first
  python3 visualizer.py --port 0 --no-browser  # print the URL only
"""

REPO = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(REPO, "viz")
TRACES = os.path.join(VIZ, "traces")


def resolve_trace(name):
    """--trace accepts a bare workload name, a filename, or a path."""
    if not name:
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
    p.add_argument("--trace", metavar="NAME", help="workload name or path of a .mtpt container to open immediately")
    p.add_argument("--build", action="store_true", help="run `make trace` first to (re)generate the containers")
    p.add_argument("--port", type=int, default=8765, help="local port to serve on (0 = pick a free one; default 8765)")
    p.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    args = p.parse_args()

    if args.build:
        build_traces()

    trace_rel = resolve_trace(args.trace)

    port = args.port or free_port()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=VIZ, **kw)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    port = httpd.server_address[1]

    url = "http://127.0.0.1:%d/index.html" % port
    if trace_rel:
        url += "?trace=" + trace_rel

    print("visualizer.py: serving %s at %s" % (VIZ, url))
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
