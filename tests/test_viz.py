import json
import time
import urllib.request

import pytest

from nfly.viz import RandomPolicy, Session, SessionConfig, VizServer, action_names_of
from nfly.suite import get_suite


@pytest.fixture
def server():
    cfg = SessionConfig(suite="classic", game="cartpole", policy="random", fps=200)
    env = get_suite("classic").make("cartpole", seed=0, render_mode="rgb_array")
    srv = VizServer(Session(cfg, env, RandomPolicy(env.action_space), action_names_of(env)), port=0).start()
    yield srv
    srv.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def _wait_for_steps(url, n=1, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = json.loads(_get(url + "/api/state")[2])
        if state["error"]:
            raise RuntimeError(state["error"])
        if state["step"] >= n:
            return state
        time.sleep(0.05)
    raise TimeoutError("streamer produced no steps")


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_page_state_and_stream(server):
    status, ctype, body = _get(server.url + "/")
    assert status == 200 and "text/html" in ctype and b"nfly viewer" in body

    state = _wait_for_steps(server.url)
    assert state["game"] == "cartpole" and state["action_names"] == ["a0", "a1"] and state["step"] > 0
    assert state["history"] and {"step", "action", "reward"} <= set(state["history"][-1])

    with urllib.request.urlopen(server.url + "/api/stream", timeout=5) as r:
        line = r.readline()
        while not line.startswith(b"data:"):
            line = r.readline()
        ev = json.loads(line[5:])
    assert ev["action_name"] in ("a0", "a1") and len(ev["frame_jpeg_b64"]) > 100 and "episode_return" in ev


def test_start_is_idempotent(server):
    assert server.start() is server


def test_controls(server):
    _wait_for_steps(server.url)
    assert _post(server.url + "/api/control", {"cmd": "pause"})["paused"] is True
    time.sleep(0.1)
    before = json.loads(_get(server.url + "/api/state")[2])["step"]
    time.sleep(0.2)
    assert json.loads(_get(server.url + "/api/state")[2])["step"] == before
    _post(server.url + "/api/control", {"cmd": "step"}); time.sleep(0.2)
    assert json.loads(_get(server.url + "/api/state")[2])["step"] == before + 1
    assert _post(server.url + "/api/control", {"cmd": "fps", "fps": 30})["fps"] == 30
    assert _post(server.url + "/api/control", {"cmd": "bogus"})["ok"] is False
    assert _post(server.url + "/api/control", {"cmd": "resume"})["paused"] is False


@pytest.fixture
def fly_server(tmp_path):
    from nfly.connectome.synthetic import write_synthetic
    from nfly.viz import build_session
    cfg = SessionConfig(suite="classic", game="cartpole", data_dir=str(write_synthetic(tmp_path)), subset="all",
                        rnn_steps=2, fps=200, max_points=200)
    srv = VizServer(build_session(cfg), port=0).start()
    yield srv
    srv.stop()


def test_anatomy_and_brain_stream(fly_server):
    import base64
    import numpy as np
    atlas = json.loads(_get(fly_server.url + "/api/anatomy")[2])
    assert 0 < atlas["n"] <= 200 and len(atlas["stages"]) == len(atlas["colors"]) == len(atlas["counts"])
    xyz = np.frombuffer(base64.b64decode(atlas["xyz"]), np.float32).reshape(-1, 3)
    assert len(xyz) == atlas["n"] and np.isfinite(xyz).all()
    assert atlas["counts"][atlas["stages"].index("descending")] == 20        # every descending neuron is kept

    _wait_for_steps(fly_server.url)
    with urllib.request.urlopen(fly_server.url + "/api/stream", timeout=5) as r:
        line = r.readline()
        while not line.startswith(b"data:"):
            line = r.readline()
        ev = json.loads(line[5:])
    brain = ev["brain"]
    cloud = np.frombuffer(base64.b64decode(brain["cloud"]), np.uint8)
    assert len(cloud) == atlas["n"] and len(brain["stages"]) == 2 and len(brain["stages"][0]) == len(atlas["stages"])
    assert all(v >= 0 for v in brain["stages"][-1]) and ev["probs"] is not None
