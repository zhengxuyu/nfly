"""Tiny standard-library web server: serves the page, a JSON state endpoint, an SSE event
stream and a control endpoint.  No framework dependency, so it runs anywhere the package does.

    GET  /                 the visualiser page
    GET  /api/state        session description + recent action history
    GET  /api/anatomy      neuron positions, stages and photoreceptor layout for the 3-D view
    GET  /api/anatomy/meshes     official neuropil meshes (when fetched by scripts/fetch_anatomy.py)
    GET  /api/anatomy/skeletons  official skeletons of the neurons that have one
    GET  /api/stream       text/event-stream of step events (frame, action, reward, probs)
    POST /api/control      {"cmd": "pause" | "resume" | "step" | "reset" | "fps", "fps": 15}
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources

from .session import Session, SessionConfig, build_session
from .streamer import Broadcast, EpisodeStreamer

PAGE = resources.files(__package__).joinpath("static/index.html").read_text(encoding="utf-8")


class VizServer:
    def __init__(self, session: Session, host: str = "127.0.0.1", port: int = 8000):
        self.broadcast = Broadcast()
        self.streamer = EpisodeStreamer(session, self.broadcast)
        handler = _make_handler(self)
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self._started = False

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "VizServer":
        if not self._started:
            self._started = True
            self.streamer.start()
            threading.Thread(target=self.httpd.serve_forever, daemon=True, name="nfly-http").start()
        return self

    def serve_forever(self) -> None:
        self.start()
        try:
            self.streamer.join()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self.streamer.stop()
        self.httpd.shutdown()
        self.httpd.server_close()

    def control(self, cmd: str, **kw) -> dict:
        s = self.streamer
        if cmd == "pause": s.pause()
        elif cmd == "resume": s.resume()
        elif cmd == "step": s.pause(); s.single_step()
        elif cmd == "reset": s.reset()
        elif cmd == "fps": s.fps = float(kw.get("fps", s.fps))
        else: return {"ok": False, "error": f"unknown cmd {cmd!r}"}
        return {"ok": True, "paused": s.paused, "fps": s.fps}


def _make_handler(server: VizServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):            # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:
            if self.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._json(server.streamer.state())
            elif self.path == "/api/anatomy":
                atlas = server.streamer.session.atlas
                self._json(atlas.to_dict() if atlas is not None else {"n": 0})
            elif self.path == "/api/anatomy/meshes":
                assets = server.streamer.session.assets
                self._json(assets.meshes_dict() if assets else {"rois": []})
            elif self.path == "/api/anatomy/skeletons":
                s = server.streamer.session
                self._json(s.assets.skeletons_dict(s.atlas) if s.assets and s.atlas else {"neurons": []})
            elif self.path == "/api/stream":
                self._stream()
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/control":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._json(server.control(body.pop("cmd", ""), **body))

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = server.broadcast.subscribe()
            try:
                while not server.streamer._stop.is_set():
                    try:
                        ev = q.get(timeout=1.0)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                        continue
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                server.broadcast.unsubscribe(q)

    return Handler


def serve(cfg: SessionConfig, host: str = "127.0.0.1", port: int = 8000) -> VizServer:
    """Standard launch flow: config -> session -> server (started, not blocking)."""
    return VizServer(build_session(cfg), host, port).start()
