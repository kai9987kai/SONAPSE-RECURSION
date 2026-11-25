"""Local research dashboard. Runs a selected study in a bounded CPU worker."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
from .runner import NAMES, run


def serve(page, out, port):
    html = page.read_bytes()
    pool = ThreadPoolExecutor(max_workers=1)
    lock = Lock()
    future = None

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def local(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.local():
                return self.reply(
                    403, {"error": "Use the loopback address printed at launch"}
                )
            if self.path == "/":
                self.reply(200, html, "text/html; charset=utf-8")
            elif self.path == "/api/status":
                with lock:
                    task = future
                if task is None or not task.done():
                    return self.reply(
                        200, {"state": "idle" if task is None else "running"}
                    )
                try:
                    result = task.result()
                    self.reply(200, {"state": "complete", "report": result})
                except Exception as e:
                    self.reply(500, {"state": "failed", "error": str(e)})
            else:
                self.reply(404, {"error": "Unknown route"})

        def do_POST(self):
            nonlocal future
            if (
                not self.local()
                or self.headers.get("Origin")
                != f"http://127.0.0.1:{self.server.server_port}"
            ):
                return self.reply(403, {"error": "Local dashboard origin required"})
            if self.path != "/api/run":
                return self.reply(404, {"error": "Unknown route"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                name, start, count = (
                    request["study"],
                    request["first"],
                    request["count"],
                )
                if (
                    name not in NAMES
                    or type(start) is not int
                    or type(count) is not int
                    or not 0 <= start <= 10000000
                    or not 1 <= count <= 16
                ):
                    raise ValueError("Choose a known study and 1–16 seeds")
            except (ValueError, TypeError, KeyError) as e:
                return self.reply(400, {"error": str(e)})
            with lock:
                if future is not None and not future.done():
                    return self.reply(
                        409, {"error": "A trial batch is already running"}
                    )
                future = pool.submit(run, [name], range(start, start + count), out, 1)
            self.reply(202, {"state": "running"})

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        finally:
            pool.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, default=Path("web/index.html"))
    parser.add_argument("--out", type=Path, default=Path("results/interactive"))
    parser.add_argument("--port", type=int, default=8789)
    args = parser.parse_args()
    serve(args.page, args.out, args.port)


if __name__ == "__main__":
    main()
