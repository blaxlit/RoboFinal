"""Web console: live map, pose, sensor data, logs, settings and commands.

Standard library only. The page (console.html) polls /api/state a few times
a second and sends commands as JSON.
"""

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from . import params as params_mod
from .evaluation import validate_ground_truth

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "console.html")
MAX_BODY = 2_000_000


def make_handler(explorer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RoboSLAM/1.0"

        def log_message(self, fmt, *args):  # keep the terminal for mission logs
            pass

        # ---- helpers --------------------------------------------------------------
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, allow_nan=False, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("request too large")
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        # ---- GET --------------------------------------------------------------------
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    with open(HTML_PATH, "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if url.path == "/api/state":
                    snap = explorer.snapshot(
                        map_version=int(q.get("map", -1)), event_id=int(q.get("ev", 0)),
                        gt_version=int(q.get("gt", 0)), param_version=int(q.get("pv", 0)),
                        traj_from=int(q.get("tf", 0)))
                    return self._send(200, _clean(snap))
                if url.path == "/api/schema":
                    return self._send(200, {"schema": params_mod.schema(), "values": dict(explorer.p)})
                if url.path == "/api/files":
                    d = explorer.run_dir
                    files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
                    return self._send(200, {"dir": d, "files": files})
                if url.path.startswith("/files/"):
                    name = os.path.basename(unquote(url.path[len("/files/"):]))
                    path = os.path.join(explorer.run_dir, name)
                    if not name or not os.path.isfile(path):
                        return self._send(404, {"error": "no such file"})
                    with open(path, "rb") as f:
                        ctype = mimetypes.guess_type(name)[0] or "text/plain"
                        if ctype.startswith("text/") or name.endswith((".csv", ".md", ".yaml")):
                            ctype = "text/plain; charset=utf-8"
                        return self._send(200, f.read(), ctype)
                return self._send(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                return self._send(500, {"error": repr(exc)})

        # ---- POST -------------------------------------------------------------------
        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = self._body()
                if url.path == "/api/cmd":
                    explorer.command(str(body.get("name", "")), body.get("args") or {})
                    return self._send(200, {"ok": True})
                if url.path == "/api/params":
                    applied = explorer.set_params(body.get("changes") or {})
                    return self._send(200, {"ok": True, "applied": applied})
                if url.path == "/api/params/save":
                    path = params_mod.save_overrides(explorer.p)
                    explorer.log(f"Settings saved to {path}")
                    return self._send(200, {"ok": True, "path": path})
                if url.path == "/api/params/defaults":
                    explorer.set_params(params_mod.defaults())
                    return self._send(200, {"ok": True})
                if url.path == "/api/gt":
                    gt = None if body.get("clear") else validate_ground_truth(body)
                    explorer.set_ground_truth(gt)
                    return self._send(200, {"ok": True})
                return self._send(404, {"error": "not found"})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, TypeError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:
                return self._send(500, {"error": repr(exc)})

    return Handler


def _clean(obj):
    """Replace NaN/inf (not valid JSON) with None."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


def serve(explorer, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(explorer))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    return server, f"http://{shown}:{port}/"
