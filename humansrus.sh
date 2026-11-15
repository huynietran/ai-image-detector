#!/usr/bin/env bash
# HumansRUS in one command (macOS / Linux). Run from anywhere:
#
#   ./humansrus.sh            start the website on this computer and open it
#   ./humansrus.sh share      same, plus a public link anyone can open while this runs
#   ./humansrus.sh test       smoke-test the website, API, detector and metrics
#   ./humansrus.sh improve    retrain with the planned improvements (needs data/matched/, ~1-2 h)
#   ./humansrus.sh adopt      switch the website to the model from the last "improve"
#
# The first run sets up Python packages in ~/aidetect (a few minutes, once).
# Double-clicking HumansRUS.command in Finder does "./humansrus.sh start".
set -euo pipefail

cd "$(dirname "$0")"
REPO="$(pwd)"
VENV="${HUMANSRUS_VENV:-$HOME/aidetect}"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"
PKGS=(fastapi "uvicorn[standard]" python-multipart httpx pillow numpy torch transformers
      scikit-learn c2pa-python pyyaml pandas diffusers peft)

say() { printf '\n\033[1;36m%s\033[0m\n' "$*"; }

setup() {
  if [ ! -x "$VENV/bin/python" ]; then
    PY=""
    for c in python3.12 python3.11 python3.10 python3; do
      if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
    done
    [ -n "$PY" ] || { echo "Python 3.10 or newer is needed: https://www.python.org/downloads/"; exit 1; }
    say "Setting up Python in $VENV (once)"
    "$PY" -m venv "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  if [ ! -f "$VENV/.humansrus-ready" ]; then
    say "Installing packages (once, a few minutes)"
    python -m pip install -q --upgrade pip
    python -m pip install -q "${PKGS[@]}"
    touch "$VENV/.humansrus-ready"
  fi
}

checkpoint() {
  python - <<'EOF'
import json, pathlib, sys
t = json.loads(pathlib.Path("evaluation/results/threshold.json").read_text())
ck = pathlib.Path(t.get("checkpoint", "model/checkpoints/step2.pt"))
if not ck.is_file():
    sys.exit(f"Missing {ck}. Download it from the team Drive (HumansRUS > HumansRUsDR > handoff_v0 > model) into {ck.parent}/.")
EOF
}

SERVER_PID=""
TUNNEL_PID=""
cleanup() {
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start_server() {
  checkpoint
  if curl -sf "$URL/health" >/dev/null 2>&1; then
    say "HumansRUS is already running at $URL"
    return
  fi
  say "Starting HumansRUS"
  python product/web/serve.py &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    curl -sf "$URL/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "The server stopped; see the messages above."; exit 1; }
    sleep 1
  done
  say "Loading the vision model (the first time downloads about 350 MB)"
  python - "$URL" <<'EOF'
import io, sys, urllib.request, uuid
from PIL import Image
buf = io.BytesIO(); Image.new("RGB", (256, 256), (120, 160, 170)).save(buf, "JPEG")
b = uuid.uuid4().hex
body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"warmup.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
req = urllib.request.Request(sys.argv[1] + "/score", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
urllib.request.urlopen(req, timeout=900).read()
print("ready")
EOF
}

cloudflared_bin() {
  if command -v cloudflared >/dev/null; then command -v cloudflared; return; fi
  local dir="$HOME/.humansrus/bin" os arch
  if [ ! -x "$dir/cloudflared" ]; then
    mkdir -p "$dir"
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    arch="$(uname -m)"; [ "$arch" = "x86_64" ] && arch=amd64; [ "$arch" = "aarch64" ] && arch=arm64
    say "Downloading cloudflared (once) for the public link"
    if [ "$os" = "darwin" ]; then
      curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-$arch.tgz" | tar -xz -C "$dir"
    else
      curl -fsSL -o "$dir/cloudflared" "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$arch"
    fi
    chmod +x "$dir/cloudflared"
  fi
  echo "$dir/cloudflared"
}

share() {
  local cf log link=""
  cf="$(cloudflared_bin)"
  log="$(mktemp)"
  "$cf" tunnel --no-autoupdate --url "$URL" >"$log" 2>&1 &
  TUNNEL_PID=$!
  for _ in $(seq 1 45); do
    link="$(grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' "$log" | head -1 || true)"
    [ -n "$link" ] && break
    sleep 1
  done
  [ -n "$link" ] || { cat "$log"; echo "Couldn't get a public link (see above). The site still works at $URL"; return; }
  command -v pbcopy >/dev/null && printf '%s' "$link" | pbcopy
  say "Public link (copied): $link"
  echo "Anyone with this link can use HumansRUS while this window stays open and this computer is awake."
  echo "Their images come to this computer to be checked. The link changes every time you start sharing."
}

open_site() {
  if command -v open >/dev/null; then open "$1"; elif command -v xdg-open >/dev/null; then xdg-open "$1"; fi
}

case "${1:-start}" in
  start)
    setup; start_server
    say "HumansRUS is running at $URL  (close this window or press Ctrl+C to stop)"
    open_site "$URL"; wait "$SERVER_PID" 2>/dev/null || sleep infinity ;;
  share)
    setup; start_server; share
    open_site "$URL"; say "Press Ctrl+C or close this window to stop sharing."
    wait "$TUNNEL_PID" 2>/dev/null || true ;;
  test)
    setup
    if python -c 'import json,pathlib,sys; t=json.loads(pathlib.Path("evaluation/results/threshold.json").read_text()); sys.exit(not pathlib.Path(t.get("checkpoint","model/checkpoints/step2.pt")).is_file())'; then
      python tests/smoke_test.py
    else
      echo "(no checkpoint yet, so testing with the dummy model)"; python tests/smoke_test.py --dummy
    fi
    python product/provenance/test_check.py
    python model/steps/smoke_test_model.py --tiny ;;
  improve)
    setup
    if command -v caffeinate >/dev/null; then caffeinate -i python model/run_improvements.py; else python model/run_improvements.py; fi ;;
  adopt)
    setup; python model/run_improvements.py --adopt ;;
  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
