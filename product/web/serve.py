"""
HumansRUS website: serves product/web/index.html and the existing API together.

Run from the repo root:
    python product/web/serve.py
Then open http://127.0.0.1:8000  (API docs stay at /docs).

It reuses product/api/main.py unchanged (same /score and /health) and adds:
  * GET /threshold  -> the flagging cut-off the page shows,
  * the page itself at /.
AI_THRESHOLD is read from evaluation/results/threshold.json unless you set it yourself.

Optional environment variables:
  CHECKPOINT_PATH   default model/checkpoints/step2.pt
  USE_DUMMY_MODEL=1 run without a checkpoint (scores are fake and labelled)
  PORT              default 8000
"""
import json
import os
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent
REPO = WEB.parents[1]
THRESHOLD_FILE = REPO / "evaluation" / "results" / "threshold.json"

if THRESHOLD_FILE.exists() and "AI_THRESHOLD" not in os.environ:
    # must be set before product.detect is imported
    os.environ["AI_THRESHOLD"] = str(json.loads(THRESHOLD_FILE.read_text())["threshold"])

os.chdir(REPO)                     # detect.py's default CHECKPOINT_PATH is relative to the repo root
sys.path.insert(0, str(REPO))

import uvicorn                                # noqa: E402
from fastapi.staticfiles import StaticFiles   # noqa: E402
from product.api.main import app              # noqa: E402


@app.get("/threshold")
def threshold():
    return {"threshold": float(os.environ.get("AI_THRESHOLD", "0.5"))}


# mounted last so /score, /health, /threshold and /docs keep priority
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    model = "dummy model (fake scores)" if os.environ.get("USE_DUMMY_MODEL") == "1" else os.environ.get("CHECKPOINT_PATH", "model/checkpoints/step2.pt")
    print(f"HumansRUS: {model} | threshold {os.environ.get('AI_THRESHOLD', '0.5')} | http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)   # this computer only
