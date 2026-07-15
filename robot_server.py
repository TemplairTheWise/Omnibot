"""
Flask server for remote OmniBot control.

Run:
    python robot_server.py

Then open http://<pi-ip>:5000 in a browser (or localhost:5000 on the Pi itself).

API
---
POST /move      {"x": <-1..1>, "y": <-1..1>, "speed": <0..100>}
POST /rotate    {"direction": "left"|"right", "speed": <0..100>}
POST /stop
POST /gripper   {"open": true|false}  or  {"toggle": true}
GET  /status    → {"gripper_closed": bool}
GET  /          → HTML control page
"""

import threading
import time

from flask import Flask, jsonify, request

from omnibot import OmniBot
from adafruit_motor import servo

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

# ── Watchdog ─────────────────────────────────────────────────────────────────
# Stops the robot if no move/rotate command arrives within WATCHDOG_TIMEOUT seconds.
# Mirrors drive.py's heartbeat behaviour but inverted: the client sends the
# heartbeat; the server stops if the client goes silent.
_lock          = threading.Lock()
_last_cmd_time = time.monotonic()


def _touch():
    global _last_cmd_time
    _last_cmd_time = time.monotonic()


def _watchdog():
    while True:
        time.sleep(0.25)
        with _lock:
            if time.monotonic() - _last_cmd_time > WATCHDOG_TIMEOUT:
                bot.stop()


threading.Thread(target=_watchdog, daemon=True).start()

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/move", methods=["POST"])
def move():
    d = request.get_json(force=True)
    x     = float(d.get("x", 0))
    y     = float(d.get("y", 0))
    speed = max(-100.0, min(100.0, float(d.get("speed", 100))))
    with _lock:
        bot.startMove([x, y], speed)
        _touch()
    return jsonify(ok=True)


@app.route("/rotate", methods=["POST"])
def rotate():
    d         = request.get_json(force=True)
    direction = "left" if d.get("direction", "left") == "left" else "right"
    speed     = max(0.0, min(100.0, float(d.get("speed", 100))))
    with _lock:
        bot.rotate(direction, speed)
        _touch()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    with _lock:
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


# ── Control page ─────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OmniBot</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #111;
    color: #eee;
    font-family: system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 28px;
    padding: 28px 16px;
    min-height: 100dvh;
    user-select: none;
  }

  h1 { font-size: 1.4rem; letter-spacing: .1em; color: #7cf; }

  /* Speed slider */
  .speed-row {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: .9rem;
  }
  input[type=range] { width: 180px; accent-color: #7cf; }

  /* D-pad grid */
  .dpad {
    display: grid;
    grid-template-columns: repeat(3, 72px);
    grid-template-rows:    repeat(3, 72px);
    gap: 6px;
  }

  /* Rotate + gripper row */
  .aux-row {
    display: flex;
    gap: 14px;
  }

  button {
    background: #222;
    border: 2px solid #444;
    border-radius: 12px;
    color: #eee;
    font-size: 1.4rem;
    cursor: pointer;
    width: 100%;
    height: 100%;
    transition: background .1s, border-color .1s;
    touch-action: none;
  }
  button:active, button.held {
    background: #2a4a6a;
    border-color: #7cf;
  }

  /* wide buttons in aux row */
  .aux-row button {
    width: 90px;
    height: 60px;
    font-size: 1rem;
  }

  .gripper-btn {
    width: 200px !important;
    font-size: .95rem !important;
  }

  .stop-btn {
    background: #3a1515;
    border-color: #c55;
    font-size: 1rem;
  }
  .stop-btn:active { background: #6a2020; border-color: #f77; }

  #log {
    font-size: .75rem;
    color: #666;
    height: 1.2em;
  }

  /* Keyboard legend */
  .kbd-legend {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 4px 12px;
    font-size: .78rem;
    color: #888;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 12px 16px;
    max-width: 260px;
    width: 100%;
  }
  .kbd-legend .label { color: #aaa; }
  kbd {
    display: inline-block;
    background: #2a2a2a;
    border: 1px solid #555;
    border-bottom-width: 2px;
    border-radius: 4px;
    padding: 1px 5px;
    font-family: monospace;
    font-size: .82rem;
    color: #ddd;
  }
</style>
</head>
<body>

<h1>OmniBot Control</h1>

<div class="speed-row">
  <span>Speed</span>
  <input type="range" id="speed" min="10" max="100" value="80">
  <span id="speedVal">80%</span>
</div>

<!-- D-pad -->
<div class="dpad">
  <!-- row 1 -->
  <div></div>
  <button id="fwd"  title="Forward">&#9650;</button>
  <div></div>
  <!-- row 2 -->
  <button id="left" title="Strafe Left">&#9668;</button>
  <button id="stp"  class="stop-btn" title="Stop">&#9632;</button>
  <button id="right" title="Strafe Right">&#9658;</button>
  <!-- row 3 -->
  <div></div>
  <button id="back" title="Backward">&#9660;</button>
  <div></div>
</div>

<!-- Rotate & gripper -->
<div class="aux-row">
  <button id="rotL" title="Rotate Left">&#8634; L</button>
  <button id="rotR" title="Rotate Right">R &#8635;</button>
</div>

<div class="aux-row">
  <button id="grip" class="gripper-btn" title="Toggle Gripper">&#x1F91A; Gripper: Open</button>
</div>

<div class="kbd-legend">
  <div><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd></div><div class="label">Move (hold)</div>
  <div><kbd>Q</kbd> / <kbd>E</kbd></div>            <div class="label">Rotate left / right (hold)</div>
  <div><kbd>G</kbd></div>                            <div class="label">Toggle gripper</div>
  <div><kbd>Space</kbd></div>                        <div class="label">Stop</div>
</div>

<div id="log"></div>

<script>
  const speedEl    = document.getElementById('speed');
  const speedLabel = document.getElementById('speedVal');
  const log        = document.getElementById('log');
  const gripBtn    = document.getElementById('grip');

  speedEl.addEventListener('input', () => {
    speedLabel.textContent = speedEl.value + '%';
  });

  let activeInterval = null;

  function post(url, body) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).catch(() => {});
  }

  function getSpeed() { return parseInt(speedEl.value, 10); }

  // Start repeating a move command every 150 ms (server watchdog = 1 s)
  function startAction(url, body) {
    stopAction();
    const payload = () => ({ ...body, speed: getSpeed() });
    post(url, payload());
    activeInterval = setInterval(() => post(url, payload()), 150);
  }

  function stopAction() {
    if (activeInterval) { clearInterval(activeInterval); activeInterval = null; }
    post('/stop', {});
  }

  // Bind a hold-to-move button
  function bindHold(id, url, body) {
    const el = document.getElementById(id);
    const start = (e) => { e.preventDefault(); el.classList.add('held'); startAction(url, body); };
    const stop  = ()  => { el.classList.remove('held'); stopAction(); };
    el.addEventListener('mousedown',   start);
    el.addEventListener('touchstart',  start, { passive: false });
    el.addEventListener('mouseup',     stop);
    el.addEventListener('mouseleave',  stop);
    el.addEventListener('touchend',    stop);
    el.addEventListener('touchcancel', stop);
  }

  // Bind a rotate hold button (rotate endpoint, not move)
  function bindRotate(id, direction) {
    const el = document.getElementById(id);
    const start = (e) => {
      e.preventDefault();
      el.classList.add('held');
      const payload = () => ({ direction, speed: getSpeed() });
      post('/rotate', payload());
      activeInterval = setInterval(() => post('/rotate', payload()), 150);
    };
    const stop = () => { el.classList.remove('held'); stopAction(); };
    el.addEventListener('mousedown',   start);
    el.addEventListener('touchstart',  start, { passive: false });
    el.addEventListener('mouseup',     stop);
    el.addEventListener('mouseleave',  stop);
    el.addEventListener('touchend',    stop);
    el.addEventListener('touchcancel', stop);
  }

  bindHold('fwd',   '/move', { x: 0,  y: 1  });
  bindHold('back',  '/move', { x: 0,  y: -1 });
  bindHold('left',  '/move', { x: -1, y: 0  });
  bindHold('right', '/move', { x: 1,  y: 0  });
  bindRotate('rotL', 'left');
  bindRotate('rotR', 'right');

  // Stop button — single press
  document.getElementById('stp').addEventListener('click', () => stopAction());

  // Gripper toggle
  let gripClosed = false;
  gripBtn.addEventListener('click', () => {
    post('/gripper', { toggle: true })
      .then(r => r && r.json())
      .then(data => {
        if (!data) return;
        gripClosed = data.closed;
        gripBtn.textContent = '\\u{1F91A} Gripper: ' + (gripClosed ? 'Closed' : 'Open');
      });
  });

  // Keyboard support (mirrors drive.py)
  const held = new Set();
  const keyMap = {
    'w': () => startAction('/move',   { x: 0,  y: 1  }),
    's': () => startAction('/move',   { x: 0,  y: -1 }),
    'a': () => startAction('/move',   { x: -1, y: 0  }),
    'd': () => startAction('/move',   { x: 1,  y: 0  }),
    'q': () => { clearInterval(activeInterval); const p = () => ({ direction: 'left',  speed: getSpeed() }); post('/rotate', p()); activeInterval = setInterval(() => post('/rotate', p()), 150); },
    'e': () => { clearInterval(activeInterval); const p = () => ({ direction: 'right', speed: getSpeed() }); post('/rotate', p()); activeInterval = setInterval(() => post('/rotate', p()), 150); },
    'g': () => gripBtn.click(),
    ' ': () => stopAction(),
  };

  document.addEventListener('keydown', (ev) => {
    const k = ev.key.toLowerCase();
    if (held.has(k)) return;
    held.add(k);
    if (keyMap[k]) { ev.preventDefault(); keyMap[k](); }
  });

  document.addEventListener('keyup', (ev) => {
    const k = ev.key.toLowerCase();
    held.delete(k);
    if (['w','s','a','d','q','e'].includes(k)) stopAction();
  });

  // Sync gripper state on load
  fetch('/status').then(r => r.json()).then(d => {
    gripClosed = d.gripper_closed;
    gripBtn.textContent = '\\u{1F91A} Gripper: ' + (gripClosed ? 'Closed' : 'Open');
  });
</script>
</body>
</html>"""


@app.route("/")
def index():
    return _HTML


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
