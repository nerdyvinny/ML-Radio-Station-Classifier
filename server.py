# the dashboard server. run it and open http://127.0.0.1:8750
#   python server.py
# it only listens on this computer (127.0.0.1), nothing is exposed to the network

import json
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import labels
import model
import radio
import sweep

PORT = 8750
HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = [sweep.freq_key(f) for f in sweep.channel_list()]

lock = threading.Lock()
state = {
    "running": False,
    "stop_wanted": False,
    "save_sweeps": True,
    "sweeps_done": 0,
    "current": None,          # index of the channel being listened to right now
    "busy": "",               # "training..." while a retrain runs
    "error": "",
    "showing": "nothing yet",
    "powers": np.full(sweep.NUM_CHANNELS, np.nan),
    "pictures": np.zeros((sweep.NUM_CHANNELS, sweep.TIME_ROWS, sweep.FFT_SIZE), dtype=np.float32),
    "probs": [None] * sweep.NUM_CHANNELS,
    "net": None,
}


def clean_number(value):
    # json can't hold NaN
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), 2)


def show_last_saved_sweep():
    # so the page isn't blank before the first live sweep
    saved = sweep.load_sweeps()
    if not saved:
        return
    last = saved[-1]
    state["powers"] = last["power_db"].copy()
    state["pictures"] = last["pictures"].copy()
    state["showing"] = "saved sweep " + last["name"]
    refresh_predictions()


def refresh_predictions():
    if state["net"] is None:
        state["probs"] = [None] * sweep.NUM_CHANNELS
        return
    probs = model.predict(state["net"], state["pictures"])
    state["probs"] = [None if np.isnan(state["powers"][i]) else probs[i].tolist()
                      for i in range(sweep.NUM_CHANNELS)]


def on_channel(i, freq, picture, power):
    # called by the sweep after every hop
    probs = None
    if state["net"] is not None:
        probs = model.predict(state["net"], picture[None])[0].tolist()
    with lock:
        state["current"] = i
        state["pictures"][i] = picture
        state["powers"][i] = power
        state["probs"][i] = probs
        state["showing"] = "live"


def sweep_worker():
    try:
        lib, dev = sweep.open_dongle()
    except Exception as problem:
        with lock:
            state["error"] = str(problem)
            state["running"] = False
        return
    try:
        while not state["stop_wanted"]:
            result = sweep.run_sweep(lib, dev, on_channel, lambda: state["stop_wanted"])
            if result is None:
                break
            if state["save_sweeps"]:
                sweep.save_sweep(result)
            with lock:
                state["sweeps_done"] += 1
    except Exception as problem:
        with lock:
            state["error"] = str(problem)
    finally:
        radio.close_radio(lib, dev)
        with lock:
            state["running"] = False
            state["current"] = None


def train_worker():
    try:
        model.train_and_report()
        net = model.load_net()
        with lock:
            state["net"] = net
            refresh_predictions()
            state["error"] = ""
    except Exception as problem:
        with lock:
            state["error"] = "training failed: " + str(problem)
    finally:
        with lock:
            state["busy"] = ""


def build_state():
    answers = labels.load_json(labels.LABELS_FILE)
    evidence = labels.load_json(labels.EVIDENCE_FILE)
    names = labels.station_names()
    with lock:
        strength = sweep.above_floor(state["powers"])
        channels = []
        for i, key in enumerate(KEYS):
            probs = state["probs"][i]
            said = None
            confidence = None
            if probs is not None:
                best = int(np.argmax(probs))
                said = model.CLASSES[best]
                confidence = clean_number(probs[best])
            answer = answers.get(key, {})
            found = evidence.get(key, {})
            channels.append({
                "freq": key,
                "power": clean_number(float(state["powers"][i])),
                "strength": clean_number(float(strength[i])),
                "probs": None if probs is None else [clean_number(p) for p in probs],
                "said": said,
                "confidence": confidence,
                "name": names[key]["name"],
                "name_source": names[key]["source"],
                "label": answer.get("label"),
                "label_source": answer.get("source"),
                "needs_review": bool(answer.get("needs_review")),
                "why": answer.get("why", ""),
                "pilot": bool(found.get("pilot")),
                "decoded": bool(found.get("decoded")),
                "has_audio": os.path.exists(os.path.join(labels.AUDIO_FOLDER, key + ".wav")),
            })
        return {
            "running": state["running"],
            "sweeps_done": state["sweeps_done"],
            "saved_sweeps": len(os.listdir(sweep.SWEEP_FOLDER)) if os.path.isdir(sweep.SWEEP_FOLDER) else 0,
            "current": state["current"],
            "busy": state["busy"],
            "error": state["error"],
            "showing": state["showing"],
            "save_sweeps": state["save_sweeps"],
            "has_model": state["net"] is not None,
            "report": model.load_report(),
            "classes": model.CLASSES,
            "channels": channels,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass    # keep the terminal quiet

    def send_bytes(self, body, content_type, status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, data, status=200):
        self.send_bytes(json.dumps(data).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            with open(os.path.join(HERE, "dashboard.html"), "rb") as f:
                self.send_bytes(f.read(), "text/html; charset=utf-8")
        elif url.path == "/api/state":
            self.send_json(build_state())
        elif url.path == "/api/picture":
            key = parse_qs(url.query).get("freq", [""])[0]
            if key not in KEYS:
                self.send_json({"error": "unknown frequency"}, 400)
                return
            with lock:
                picture = state["pictures"][KEYS.index(key)]
                self.send_json({"picture": np.round(picture, 3).tolist()})
        elif url.path.startswith("/audio/"):
            key = url.path[len("/audio/"):].replace(".wav", "")
            # only ever open a file whose name is one of our 101 frequencies
            if key not in KEYS:
                self.send_json({"error": "unknown frequency"}, 400)
                return
            path = os.path.join(labels.AUDIO_FOLDER, key + ".wav")
            if not os.path.exists(path):
                self.send_json({"error": "no recording for this channel"}, 404)
                return
            with open(path, "rb") as f:
                self.send_bytes(f.read(), "audio/wav")
        else:
            self.send_json({"error": "not found"}, 404)

    def read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0 or length > 10000:
            return {}
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    def do_POST(self):
        # only our own page may press the buttons, not some other website
        origin = self.headers.get("Origin")
        allowed = ["http://127.0.0.1:" + str(PORT), "http://localhost:" + str(PORT)]
        if origin is not None and origin not in allowed:
            self.send_json({"error": "not allowed"}, 403)
            return
        body = self.read_body()
        if self.path == "/api/start":
            with lock:
                if state["running"]:
                    self.send_json({"ok": True})
                    return
                state["running"] = True
                state["stop_wanted"] = False
                state["error"] = ""
                state["save_sweeps"] = bool(body.get("save", True))
            threading.Thread(target=sweep_worker, daemon=True).start()
            self.send_json({"ok": True})
        elif self.path == "/api/stop":
            state["stop_wanted"] = True
            self.send_json({"ok": True})
        elif self.path == "/api/label":
            key = body.get("freq")
            answer = body.get("label")
            if key not in KEYS or answer not in model.CLASSES:
                self.send_json({"error": "bad frequency or label"}, 400)
                return
            with lock:
                answers = labels.load_json(labels.LABELS_FILE)
                answers[key] = {"label": answer, "source": "you", "needs_review": False,
                                "why": "you listened and decided"}
                labels.save_json(labels.LABELS_FILE, answers)
            self.send_json({"ok": True})
        elif self.path == "/api/retrain":
            with lock:
                if state["busy"]:
                    self.send_json({"ok": True})
                    return
                state["busy"] = "training the net (about a minute)..."
            threading.Thread(target=train_worker, daemon=True).start()
            self.send_json({"ok": True})
        else:
            self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    state["net"] = model.load_net()
    show_last_saved_sweep()
    print("dashboard at http://127.0.0.1:" + str(PORT) + "  (Ctrl+C to quit)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
