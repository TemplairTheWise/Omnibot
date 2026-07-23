"""
Flask server for remote OmniBot control.

Run:
    python robot_server.py

Then open http://<pi-ip>:5000 in a browser (or localhost:5000 on the Pi itself).

API
---
POST /move              {"x": <-1..1>, "y": <-1..1>, "speed": <0..100>}
POST /rotate            {"direction": "left"|"right", "speed": <0..100>}
POST /stop
POST /gripper           {"open": true|false}  or  {"toggle": true}
GET  /status            → {"gripper_closed": bool}
GET  /                  → HTML control page

POST /search/start      {"target": "bottle-plastic", "scan": false, "record": "out.mp4", "flip": true}
POST /search/stop
GET  /search/status     → {"state": "APPROACHING", "target": ..., "detected": ..., ...}
GET  /search/labels     → ["bottle-plastic", ...]
GET  /search/history    → [{session_id, target, start_time, duration_s, outcome, detections_count}, ...]
"""

import logging
import threading
import time

from flask import Flask, jsonify, request

from omnibot import OmniBot
from adafruit_motor import servo

log = logging.getLogger(__name__)

# ── Tuning ──────────────────────────────────────────────────────────────────
GRIPPER_OPEN_ANGLE   = 0
GRIPPER_CLOSED_ANGLE = 150
GRIPPER_CHANNEL      = 7
WATCHDOG_TIMEOUT     = 1.0   # seconds of silence before auto-stop

# ── Hardware init ────────────────────────────────────────────────────────────
bot             = OmniBot()
gripper_servo   = servo.Servo(bot.pca.channels[GRIPPER_CHANNEL])
gripper_closed  = False
gripper_servo.angle = GRIPPER_OPEN_ANGLE

# ── Navigation pipeline (optional) ──────────────────────────────────────────
# If Hailo hardware is not present the server still starts for manual control.
_pipeline       = None
_sonar          = None
_nsm            = None
_nav_labels     = []

try:
    from inference_pipeline import InferencePipeline, BEVERAGE_LABELS
    from sonar_guard import SonarGuard
    from state_machine import NavStateMachine

    _pipeline   = InferencePipeline()
    _pipeline.start()
    _sonar      = SonarGuard()
    _sonar.start()
    _nsm        = NavStateMachine(bot, _pipeline, sonar=_sonar, skip_scan=True)
    _nav_labels = list(BEVERAGE_LABELS)
    log.info("Navigation pipeline ready (%d labels).", len(_nav_labels))
except Exception as _nav_exc:
    log.warning("Navigation unavailable: %s", _nav_exc)

# ── Watchdog ─────────────────────────────────────────────────────────────────
# Stops the robot if no move/rotate command arrives within WATCHDOG_TIMEOUT seconds.
# Suppressed while autonomous navigation is active (NSM state != IDLE).
_lock          = threading.Lock()
_last_cmd_time = time.monotonic()


def _touch():
    global _last_cmd_time
    _last_cmd_time = time.monotonic()


def _nsm_is_active() -> bool:
    """True when the nav state machine is running and should own the bot."""
    if _nsm is None:
        return False
    return _nsm.get_status()["state"] not in ("IDLE", "FOUND")


def _watchdog():
    while True:
        time.sleep(0.25)
        with _lock:
            if _nsm_is_active():
                _touch()   # keep timer from firing during autonomous run
            elif time.monotonic() - _last_cmd_time > WATCHDOG_TIMEOUT:
                bot.stop()


threading.Thread(target=_watchdog, daemon=True).start()

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


def _stop_nsm_if_active():
    """Stop autonomous navigation before processing a manual drive command."""
    if _nsm_is_active():
        _nsm.stop()


@app.route("/move", methods=["POST"])
def move():
    d = request.get_json(force=True)
    x     = float(d.get("x", 0))
    y     = float(d.get("y", 0))
    speed = max(-100.0, min(100.0, float(d.get("speed", 100))))
    with _lock:
        _stop_nsm_if_active()
        bot.startMove([x, y], speed)
        _touch()
    return jsonify(ok=True)


@app.route("/rotate", methods=["POST"])
def rotate():
    d         = request.get_json(force=True)
    direction = "left" if d.get("direction", "left") == "left" else "right"
    speed     = max(0.0, min(100.0, float(d.get("speed", 100))))
    with _lock:
        _stop_nsm_if_active()
        bot.rotate(direction, speed)
        _touch()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    with _lock:
        if _nsm is not None:
            _nsm.stop()
        bot.stop()
        _touch()
    return jsonify(ok=True)


@app.route("/gripper", methods=["POST"])
def gripper():
    global gripper_closed
    d = request.get_json(force=True)
    if d.get("toggle"):
        gripper_closed = not gripper_closed
    elif "open" in d:
        gripper_closed = not bool(d["open"])
    gripper_servo.angle = GRIPPER_CLOSED_ANGLE if gripper_closed else GRIPPER_OPEN_ANGLE
    return jsonify(ok=True, closed=gripper_closed)


@app.route("/status")
def status():
    return jsonify(gripper_closed=gripper_closed)


# ── Search / autonomous navigation routes ────────────────────────────────────

@app.route("/search/start", methods=["POST"])
def search_start():
    if _nsm is None:
        return jsonify(ok=False, error="navigation hardware not available"), 503

    d          = request.get_json(force=True) or {}
    target     = d.get("target") or None
    do_scan    = bool(d.get("scan", False))
    record     = d.get("record") or None
    flip       = bool(d.get("flip", True))

    ok = _nsm.start(
        target_label=target,
        record_path=record,
        record_flip=flip,
        skip_scan=not do_scan,
    )
    if not ok:
        return jsonify(ok=False, error="already running"), 409
    log.info("Search started — target=%s scan=%s record=%s", target, do_scan, record)
    return jsonify(ok=True)


@app.route("/search/stop", methods=["POST"])
def search_stop():
    if _nsm is None:
        return jsonify(ok=False, error="navigation hardware not available"), 503
    _nsm.stop()
    return jsonify(ok=True)


@app.route("/search/status")
def search_status():
    if _nsm is None:
        return jsonify(state="unavailable")
    return jsonify(_nsm.get_status())


@app.route("/search/labels")
def search_labels():
    return jsonify(_nav_labels)


@app.route("/search/history")
def search_history():
    if _nsm is None:
        return jsonify([])
    limit = min(int(request.args.get("limit", 20)), 100)
    return jsonify(_nsm.get_history(limit=limit))


# ── Control page ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniBot</title>
<style>
/* ─── Tokens ───────────────────────────────────────────────────────────── */
:root {
  --bg:          #050a05;
  --surf:        #091009;
  --raised:      #0e180e;
  --border:      #1c2e1c;
  --text:        #b8d8b8;
  --muted:       #486048;
  --accent:      #39ff84;
  --a-glow:      rgba(57,255,132,.35);
  --a-dim:       rgba(57,255,132,.08);
  --a-mid:       rgba(57,255,132,.18);
  --danger:      #ff3a3a;
  --d-glow:      rgba(255,58,58,.35);
  --d-dim:       rgba(255,58,58,.1);
  --warn:        #ffb030;
  --blue:        #60aaff;
  --b-glow:      rgba(60,140,255,.35);
  --mono: ui-monospace,'Cascadia Code','Fira Mono',monospace;
  --sans: system-ui,-apple-system,sans-serif;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg:    #edf3ed;
    --surf:  #f6faf6;
    --raised:#deeade;
    --border:#aecaae;
    --text:  #162016;
    --muted: #567056;
    --accent:#18883c;
    --a-glow:rgba(24,136,60,.25);
    --a-dim: rgba(24,136,60,.08);
    --a-mid: rgba(24,136,60,.14);
    --danger:#c42020;
    --d-glow:rgba(196,32,32,.25);
    --d-dim: rgba(196,32,32,.08);
    --blue:  #2060cc;
    --b-glow:rgba(32,96,204,.25);
  }
}
:root[data-theme="dark"] {
  --bg:#050a05; --surf:#091009; --raised:#0e180e; --border:#1c2e1c;
  --text:#b8d8b8; --muted:#486048;
  --accent:#39ff84; --a-glow:rgba(57,255,132,.35); --a-dim:rgba(57,255,132,.08); --a-mid:rgba(57,255,132,.18);
  --danger:#ff3a3a; --d-glow:rgba(255,58,58,.35); --d-dim:rgba(255,58,58,.1);
  --warn:#ffb030; --blue:#60aaff; --b-glow:rgba(60,140,255,.35);
}
:root[data-theme="light"] {
  --bg:#edf3ed; --surf:#f6faf6; --raised:#deeade; --border:#aecaae;
  --text:#162016; --muted:#567056;
  --accent:#18883c; --a-glow:rgba(24,136,60,.25); --a-dim:rgba(24,136,60,.08); --a-mid:rgba(24,136,60,.14);
  --danger:#c42020; --d-glow:rgba(196,32,32,.25); --d-dim:rgba(196,32,32,.08);
  --blue:#2060cc; --b-glow:rgba(32,96,204,.25);
}

/* ─── Reset ────────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:focus-visible { outline: 1px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }

/* ─── Shell ────────────────────────────────────────────────────────────── */
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 15px;
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  position: relative;
}
/* CRT scanline texture — nearly invisible, adds depth */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    180deg,
    transparent 0px, transparent 3px,
    rgba(0,255,80,.016) 3px, rgba(0,255,80,.016) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ─── Top bar ──────────────────────────────────────────────────────────── */
.topbar {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: stretch;
  background: var(--surf);
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.topbar-title {
  padding: 0 18px;
  display: flex;
  align-items: center;
  font-family: var(--mono);
  font-size: .65rem;
  font-weight: 700;
  letter-spacing: .25em;
  color: var(--accent);
  text-shadow: 0 0 14px var(--a-glow);
  border-right: 1px solid var(--border);
  white-space: nowrap;
  flex-shrink: 0;
}
.tab-bar { display: flex; }
.tab {
  padding: 0 22px;
  height: 48px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  color: var(--muted);
  font-family: var(--mono);
  font-size: .7rem;
  font-weight: 600;
  letter-spacing: .12em;
  text-transform: uppercase;
  cursor: pointer;
  transition: color .15s, border-color .15s, text-shadow .2s;
  white-space: nowrap;
}
.tab:hover { color: var(--text); }
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  text-shadow: 0 0 10px var(--a-glow);
}

/* ─── Panels ───────────────────────────────────────────────────────────── */
.panel {
  display: none;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  padding: 22px 16px 52px;
  flex: 1;
}
.panel.visible { display: flex; }

/* ─── State card ───────────────────────────────────────────────────────── */
/* Wrapper carries the glow; inner card carries the visual border + content */
.state-outer {
  position: relative;
  width: 100%;
  max-width: 360px;
  transition: box-shadow .5s;
  box-shadow: 0 0 6px rgba(57,255,132,.08);
}
.state-outer.si-SCANNING    { box-shadow: 0 0 18px var(--b-glow); }
.state-outer.si-SEARCHING   { box-shadow: 0 0 18px rgba(255,176,48,.4); }
.state-outer.si-APPROACHING { box-shadow: 0 0 22px var(--a-glow); }
.state-outer.si-FOUND       { animation: found-pulse 1.1s ease-in-out infinite; }
.state-outer.si-unavailable { box-shadow: 0 0 18px var(--d-glow); }

@keyframes found-pulse {
  0%,100% { box-shadow: 0 0 18px var(--a-glow); }
  50%      { box-shadow: 0 0 32px rgba(100,255,160,.7); }
}

.state-card {
  background: var(--surf);
  border: 1px solid var(--border);
  overflow: hidden;
  /* Diagonal cut: top-right corner */
  clip-path: polygon(0 0, calc(100% - 20px) 0, 100% 20px, 100% 100%, 0 100%);
}

/* Corner accent line — sits outside clip area as a sibling, not child */
.corner-mark {
  position: absolute;
  top: 0; right: 0;
  width: 20px; height: 20px;
  /* fill with page bg to "cut" the corner */
  background: var(--bg);
  clip-path: polygon(0 0, 100% 0, 100% 100%);
  pointer-events: none;
  z-index: 2;
}
/* diagonal accent line in the cut */
.corner-mark::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(-45deg,
    transparent 0%, transparent 40%,
    var(--accent) 40%, var(--accent) 60%,
    transparent 60%);
  opacity: .45;
}

/* State header */
.state-header {
  padding: 18px 20px 12px;
  border-bottom: 1px solid var(--border);
  transition: background .4s;
}
.si-IDLE        .state-header { background: transparent; }
.si-SCANNING    .state-header { background: rgba(30,80,200,.1); }
.si-SEARCHING   .state-header { background: rgba(200,120,0,.1); }
.si-APPROACHING .state-header { background: rgba(30,180,80,.1); }
.si-FOUND       .state-header { background: rgba(30,200,90,.18); }
.si-unavailable .state-header { background: rgba(200,40,40,.1); }

.state-name {
  font-family: var(--mono);
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: .1em;
  margin-bottom: 4px;
  transition: color .4s, text-shadow .4s;
}
.si-IDLE        .state-name { color: #486048; text-shadow: none; }
.si-SCANNING    .state-name { color: var(--blue);   text-shadow: 0 0 14px var(--b-glow); }
.si-SEARCHING   .state-name { color: var(--warn);   text-shadow: 0 0 14px rgba(255,176,48,.5); }
.si-APPROACHING .state-name { color: var(--accent); text-shadow: 0 0 16px var(--a-glow); }
.si-FOUND       .state-name { color: #7fffa8;       text-shadow: 0 0 20px rgba(100,255,160,.7); }
.si-unavailable .state-name { color: var(--danger); text-shadow: 0 0 12px var(--d-glow); }

.state-desc {
  font-family: var(--mono);
  font-size: .7rem;
  color: var(--muted);
  letter-spacing: .06em;
}

/* Telemetry grid */
.telem-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
}
.telem-cell {
  padding: 11px 18px;
  border-bottom: 1px solid var(--border);
  border-right: 1px solid var(--border);
}
.telem-cell:nth-child(even)      { border-right: none; }
.telem-cell:nth-last-child(-n+2) { border-bottom: none; }
.telem-key {
  font-family: var(--mono);
  font-size: .6rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 4px;
}
.telem-val {
  font-family: var(--mono);
  font-size: .88rem;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ─── Controls card ────────────────────────────────────────────────────── */
.ctrl-card {
  width: 100%;
  max-width: 360px;
  background: var(--surf);
  border: 1px solid var(--border);
  /* Smaller cut on bottom-left */
  clip-path: polygon(0 0, 100% 0, 100% 100%, 12px 100%, 0 calc(100% - 12px));
  padding: 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.field-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
}
.field-key {
  font-family: var(--mono);
  font-size: .6rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
  min-width: 52px;
}
select, input[type=text] {
  flex: 1;
  min-width: 100px;
  background: var(--raised);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 7px 10px;
  font-family: var(--mono);
  font-size: .78rem;
  transition: border-color .15s, box-shadow .15s;
}
select:focus, input[type=text]:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 2px var(--a-dim);
}
.toggle-lbl {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--mono);
  font-size: .68rem;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
}
input[type=checkbox] { accent-color: var(--accent); cursor: pointer; }

/* ─── Action buttons ───────────────────────────────────────────────────── */
.action-row { display: flex; gap: 10px; }

.btn {
  flex: 1;
  height: 50px;
  font-family: var(--mono);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  cursor: pointer;
  border-radius: 2px;
  border: 1px solid transparent;
  transition: background .12s, box-shadow .15s, border-color .12s, color .12s;
  touch-action: none;
}

/* START — filled solid, dominant primary action */
.btn-start {
  background: var(--accent);
  color: #040c04;
  border-color: var(--accent);
}
.btn-start:hover {
  box-shadow: 0 0 20px var(--a-glow);
}
.btn-start.running {
  animation: start-pulse 1.4s ease-in-out infinite;
}
@keyframes start-pulse {
  0%,100% { box-shadow: 0 0 8px var(--a-glow); }
  50%      { box-shadow: 0 0 24px var(--a-glow); }
}

/* STOP — outlined, clearly distinct, fills red on hover */
.btn-stop {
  background: transparent;
  color: var(--danger);
  border-color: var(--danger);
  opacity: .75;
}
.btn-stop:hover {
  background: var(--d-dim);
  box-shadow: 0 0 14px var(--d-glow);
  opacity: 1;
}

#log {
  font-family: var(--mono);
  font-size: .68rem;
  color: var(--muted);
  min-height: 1.2em;
  letter-spacing: .04em;
}

/* ─── Manual panel ─────────────────────────────────────────────────────── */
.speed-row {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
  max-width: 240px;
}
.speed-lbl {
  font-family: var(--mono);
  font-size: .6rem;
  font-weight: 700;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--muted);
}
input[type=range] { flex: 1; accent-color: var(--accent); }
#speedVal {
  font-family: var(--mono);
  font-size: .78rem;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  min-width: 36px;
}

.dpad {
  display: grid;
  grid-template-columns: repeat(3, 68px);
  grid-template-rows:    repeat(3, 68px);
  gap: 5px;
}
.dpad-btn {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--text);
  font-size: 1.2rem;
  cursor: pointer;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background .08s, border-color .1s, box-shadow .1s, color .08s;
  touch-action: none;
}
.dpad-btn:active, .dpad-btn.held {
  background: var(--a-dim);
  border-color: var(--accent);
  box-shadow: 0 0 10px var(--a-glow);
  color: var(--accent);
}
.dpad-stp {
  background: var(--d-dim);
  border-color: rgba(255,58,58,.35);
  color: var(--danger);
  font-family: var(--mono);
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .1em;
}
.dpad-stp:active {
  background: rgba(255,58,58,.18);
  border-color: var(--danger);
  box-shadow: 0 0 10px var(--d-glow);
}

.aux-row { display: flex; gap: 8px; }
.aux-btn {
  width: 86px;
  height: 46px;
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 2px;
  color: var(--text);
  font-family: var(--mono);
  font-size: .72rem;
  font-weight: 600;
  letter-spacing: .08em;
  cursor: pointer;
  transition: background .08s, border-color .1s, box-shadow .1s, color .08s;
  touch-action: none;
}
.aux-btn:active, .aux-btn.held {
  background: var(--a-dim);
  border-color: var(--accent);
  box-shadow: 0 0 10px var(--a-glow);
  color: var(--accent);
}
.grip-btn { width: 180px; }

.kbd-legend {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 6px 16px;
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 12px 16px;
  width: 100%;
  max-width: 240px;
}
.kl-key  { font-family: var(--mono); font-size: .7rem; color: var(--text); }
.kl-desc { font-family: var(--mono); font-size: .68rem; color: var(--muted); letter-spacing: .05em; }
kbd {
  display: inline-block;
  background: var(--raised);
  border: 1px solid var(--border);
  border-bottom-width: 2px;
  border-radius: 2px;
  padding: 1px 5px;
  font-family: var(--mono);
  font-size: .68rem;
  color: var(--text);
}
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div class="topbar-title">OMNIBOT</div>
  <nav class="tab-bar" role="tablist">
    <button class="tab active" data-tab="search" role="tab" aria-selected="true">Autonomous</button>
    <button class="tab"        data-tab="manual" role="tab" aria-selected="false">Manual</button>
  </nav>
</div>

<!-- ── Autonomous panel ──────────────────────────────────────────────────── -->
<div class="panel visible" id="panel-search" role="tabpanel">

  <!-- State card -->
  <div class="state-outer si-IDLE" id="stateOuter">
    <div class="state-card" id="stateCard">
      <div class="state-header" id="stateHeader">
        <div class="state-name" id="stateName">IDLE</div>
        <div class="state-desc" id="stateDesc">Waiting for a search command</div>
      </div>
      <div class="telem-grid">
        <div class="telem-cell">
          <div class="telem-key">Target</div>
          <div class="telem-val" id="svTarget">—</div>
        </div>
        <div class="telem-cell">
          <div class="telem-key">Detected</div>
          <div class="telem-val" id="svDetected">—</div>
        </div>
        <div class="telem-cell">
          <div class="telem-key">Bearing</div>
          <div class="telem-val" id="svBearing">—</div>
        </div>
        <div class="telem-cell">
          <div class="telem-key">Distance</div>
          <div class="telem-val" id="svSonar">—</div>
        </div>
      </div>
    </div>
    <!-- Corner cut accent — sibling of state-card so it sits above the clip -->
    <div class="corner-mark"></div>
  </div>

  <!-- Controls -->
  <div class="ctrl-card">
    <div class="field-row">
      <span class="field-key">Target</span>
      <select id="searchTarget">
        <option value="">Any beverage</option>
      </select>
    </div>
    <div class="field-row">
      <span class="field-key">Record</span>
      <input type="text" id="recordPath" placeholder="debug.mp4 (optional)">
      <label class="toggle-lbl"><input type="checkbox" id="doFlip" checked>&nbsp;Flip</label>
    </div>
    <div class="field-row">
      <span class="field-key">Options</span>
      <label class="toggle-lbl"><input type="checkbox" id="doScan">&nbsp;Polar scan first</label>
    </div>
    <div class="action-row">
      <button class="btn btn-start" id="searchStart">Start search</button>
      <button class="btn btn-stop"  id="searchStop">Stop</button>
    </div>
  </div>

  <div id="log"></div>
</div>

<!-- ── Manual panel ──────────────────────────────────────────────────────── -->
<div class="panel" id="panel-manual" role="tabpanel">

  <div class="speed-row">
    <span class="speed-lbl">Speed</span>
    <input type="range" id="speed" min="10" max="100" value="80">
    <span id="speedVal">80%</span>
  </div>

  <div class="dpad">
    <div></div>
    <button class="dpad-btn" id="fwd"   title="Forward">&#9650;</button>
    <div></div>
    <button class="dpad-btn" id="left"  title="Strafe left">&#9668;</button>
    <button class="dpad-btn dpad-stp" id="stp" title="Stop">STOP</button>
    <button class="dpad-btn" id="right" title="Strafe right">&#9658;</button>
    <div></div>
    <button class="dpad-btn" id="back"  title="Backward">&#9660;</button>
    <div></div>
  </div>

  <div class="aux-row">
    <button class="aux-btn" id="rotL">&#8634; Left</button>
    <button class="aux-btn" id="rotR">Right &#8635;</button>
  </div>
  <div class="aux-row">
    <button class="aux-btn grip-btn" id="grip">Gripper: Open</button>
  </div>

  <div class="kbd-legend">
    <div class="kl-key"><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></div>
    <div class="kl-desc">Move (hold)</div>
    <div class="kl-key"><kbd>Q</kbd>&thinsp;/&thinsp;<kbd>E</kbd></div>
    <div class="kl-desc">Rotate</div>
    <div class="kl-key"><kbd>G</kbd></div>
    <div class="kl-desc">Gripper</div>
    <div class="kl-key"><kbd>Space</kbd></div>
    <div class="kl-desc">Stop</div>
  </div>
</div>

<script>
  // ── Tab switching ──────────────────────────────────────────────────────────
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected','false'); });
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('visible'));
      tab.classList.add('active');
      tab.setAttribute('aria-selected','true');
      document.getElementById('panel-' + tab.dataset.tab).classList.add('visible');
    });
  });

  // ── Shared ─────────────────────────────────────────────────────────────────
  function post(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).catch(() => {});
  }

  const speedEl = document.getElementById('speed');
  const logEl   = document.getElementById('log');
  speedEl.addEventListener('input', () => {
    document.getElementById('speedVal').textContent = speedEl.value + '%';
  });
  function getSpeed() { return parseInt(speedEl.value, 10); }

  // ── Manual drive ───────────────────────────────────────────────────────────
  let activeInterval = null;

  function startAction(url, body) {
    stopAction();
    const pl = () => ({ ...body, speed: getSpeed() });
    post(url, pl());
    activeInterval = setInterval(() => post(url, pl()), 150);
  }

  function stopAction() {
    if (activeInterval) { clearInterval(activeInterval); activeInterval = null; }
    post('/stop', {});
  }

  function bindHold(id, url, body) {
    const el = document.getElementById(id);
    const s  = e => { e.preventDefault(); el.classList.add('held'); startAction(url, body); };
    const en = () => { el.classList.remove('held'); stopAction(); };
    ['mousedown','touchstart'].forEach(ev => el.addEventListener(ev, s, { passive: false }));
    ['mouseup','mouseleave','touchend','touchcancel'].forEach(ev => el.addEventListener(ev, en));
  }

  function bindRotate(id, dir) {
    const el = document.getElementById(id);
    const s  = e => {
      e.preventDefault(); el.classList.add('held');
      const pl = () => ({ direction: dir, speed: getSpeed() });
      post('/rotate', pl());
      activeInterval = setInterval(() => post('/rotate', pl()), 150);
    };
    const en = () => { el.classList.remove('held'); stopAction(); };
    ['mousedown','touchstart'].forEach(ev => el.addEventListener(ev, s, { passive: false }));
    ['mouseup','mouseleave','touchend','touchcancel'].forEach(ev => el.addEventListener(ev, en));
  }

  bindHold('fwd',   '/move', { x: 0,  y:  1 });
  bindHold('back',  '/move', { x: 0,  y: -1 });
  bindHold('left',  '/move', { x: -1, y:  0 });
  bindHold('right', '/move', { x:  1, y:  0 });
  bindRotate('rotL', 'left');
  bindRotate('rotR', 'right');
  document.getElementById('stp').addEventListener('click', () => stopAction());

  // Gripper
  let gripClosed = false;
  const gripBtn  = document.getElementById('grip');
  gripBtn.addEventListener('click', () => {
    post('/gripper', { toggle: true }).then(r => r && r.json()).then(d => {
      if (!d) return;
      gripClosed = d.closed;
      gripBtn.textContent = 'Gripper: ' + (gripClosed ? 'Closed' : 'Open');
    });
  });
  fetch('/status').then(r => r.json()).then(d => {
    gripClosed = d.gripper_closed;
    gripBtn.textContent = 'Gripper: ' + (gripClosed ? 'Closed' : 'Open');
  });

  // Keyboard
  const held = new Set();
  const keyMap = {
    'w': () => startAction('/move', { x: 0,  y:  1 }),
    's': () => startAction('/move', { x: 0,  y: -1 }),
    'a': () => startAction('/move', { x: -1, y:  0 }),
    'd': () => startAction('/move', { x:  1, y:  0 }),
    'q': () => { clearInterval(activeInterval); const p = () => ({ direction: 'left',  speed: getSpeed() }); post('/rotate', p()); activeInterval = setInterval(() => post('/rotate', p()), 150); },
    'e': () => { clearInterval(activeInterval); const p = () => ({ direction: 'right', speed: getSpeed() }); post('/rotate', p()); activeInterval = setInterval(() => post('/rotate', p()), 150); },
    'g': () => gripBtn.click(),
    ' ': () => stopAction(),
  };
  document.addEventListener('keydown', ev => {
    const k = ev.key.toLowerCase();
    if (held.has(k)) return;
    held.add(k);
    if (keyMap[k]) { ev.preventDefault(); keyMap[k](); }
  });
  document.addEventListener('keyup', ev => {
    const k = ev.key.toLowerCase();
    held.delete(k);
    if ('wsadqe'.includes(k)) stopAction();
  });

  // ── Autonomous search ──────────────────────────────────────────────────────
  const STATE_META = {
    IDLE:        { desc: 'Waiting for a search command' },
    SCANNING:    { desc: 'Rotating 360° to map surroundings' },
    SEARCHING:   { desc: 'Navigating toward target area' },
    APPROACHING: { desc: 'Target acquired — closing in' },
    FOUND:       { desc: 'Target reached' },
    unavailable: { desc: 'Navigation hardware offline' },
  };

  const stateOuter = document.getElementById('stateOuter');
  const stateName  = document.getElementById('stateName');
  const stateDesc  = document.getElementById('stateDesc');
  const startBtn   = document.getElementById('searchStart');

  fetch('/search/labels').then(r => r.json()).then(labels => {
    const sel = document.getElementById('searchTarget');
    labels.forEach(lbl => {
      const o = document.createElement('option');
      o.value = lbl; o.textContent = lbl;
      sel.appendChild(o);
    });
  }).catch(() => {});

  document.getElementById('searchStart').addEventListener('click', () => {
    const target = document.getElementById('searchTarget').value || null;
    const scan   = document.getElementById('doScan').checked;
    const record = document.getElementById('recordPath').value.trim() || null;
    const flip   = document.getElementById('doFlip').checked;
    post('/search/start', { target, scan, record, flip })
      .then(r => r && r.json())
      .then(d => { if (d && !d.ok) logEl.textContent = d.error || 'start failed'; });
  });

  document.getElementById('searchStop').addEventListener('click', () => post('/search/stop', {}));

  function updateStatus() {
    fetch('/search/status').then(r => r.json()).then(d => {
      const state = d.state || 'unavailable';
      const meta  = STATE_META[state] || { desc: state };

      stateOuter.className = 'state-outer si-' + state;
      stateName.textContent = state;
      stateDesc.textContent = meta.desc;

      document.getElementById('svTarget').textContent =
        d.target || '—';
      const conf = d.confidence != null ? ' ' + Math.round(d.confidence * 100) + '%' : '';
      document.getElementById('svDetected').textContent =
        d.detected ? d.detected + conf : '—';
      document.getElementById('svBearing').textContent =
        d.bearing_deg != null ? d.bearing_deg.toFixed(1) + '°' : '—';
      document.getElementById('svSonar').textContent =
        d.sonar_cm != null ? d.sonar_cm.toFixed(0) + ' cm' : '—';

      const running = !['IDLE', 'FOUND', 'unavailable'].includes(state);
      startBtn.classList.toggle('running', running);
      startBtn.textContent = running ? 'Running…' : 'Start search';
    }).catch(() => {});
  }

  updateStatus();
  setInterval(updateStatus, 1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return _HTML


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=5000, threaded=True)
