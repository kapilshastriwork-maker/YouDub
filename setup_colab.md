# YouDub — Colab Runner Notebook

This notebook is a **thin runner**. All actual setup logic lives in
`scripts/colab_setup.py` in the repo, so fixes only need to happen in one
place (the repo). Re-running cells is safe; the setup script is idempotent.

Recommended runtime: **T4 GPU**, **High RAM**, **~15 GB free Drive space**.

---

## Two-environment architecture

The YouDub pipeline runs in **two separate Python environments** because
Colab's default kernel (Python 3.13) has no wheel for CosyVoice3's exact
pins (torch 2.3.1, numpy 1.26.4, onnxruntime-gpu 1.18.0,
openai-whisper 20231117). Rather than relax pins (which risks Qwen2 model
breakage) or pin Colab to Py3.12 (no UI option), the setup script isolates
CosyVoice in its own Python 3.11 venv on Drive.

| Environment | Python | Where | Used for |
|---|---|---|---|
| **Main Colab kernel** | 3.13 (Colab default) | The Colab notebook process | `pipeline.py` orchestration, faster-whisper, Ollama, ffmpeg, all 8 pipeline functions except TTS |
| **CosyVoice venv** | 3.11 (apt-installed) | `$DRIVE_ROOT/cosyv_venv311` | CosyVoice's `AutoModel` + TTS inference via `scripts/cosyv_infer.py` subprocess |

The two environments communicate over a newline-delimited JSON protocol on
stdin/stdout. The main kernel spawns one long-running CosyVoice subprocess
per `synthesize_dubbed_audio` call, sends one job per segment, and reads
one result per segment. The 30-40 s model load is paid once and amortized
across all segments.

The venv lives on Drive so it persists across Colab reconnects. You only
pay the install cost once per Drive account.

---

> **Deprecated:** The previous 11 inline setup cells are gone. The
> previous "filter onnxruntime / openai-whisper in the main env" approach
> is also gone — it was a workaround for a Python 3.13 compatibility
> problem that the venv approach solves cleanly. If you're following a
> tutorial or older doc that references them, ignore it — the script and
> the venv are the source of truth. See `progress.md` → "## Python 3.13
> compatibility" for the full history.

---

## Cell 1 — Clone or update the YouDub repo

```python
import os, subprocess
PROJ = "/content/drive/MyDrive/YouDub"   # YouDub project root on Drive
if not os.path.isdir(f"{PROJ}/.git"):
    print(f"Cloning YouDub into {PROJ} (one-time)...")
    os.makedirs(PROJ, exist_ok=True)
    subprocess.check_call([
        "git", "clone",
        "https://github.com/kapilshastriwork-maker/YouDub.git", PROJ,
    ])
else:
    print(f"Pulling latest YouDub into {PROJ}...")
    subprocess.check_call(["git", "-C", PROJ, "pull", "--rebase"])
# Print last commit so you can confirm you got the version you expected.
print(subprocess.check_output(["git", "-C", PROJ, "log", "-1", "--oneline"],
                              text=True).strip())
```

---

## Cell 2 — Run the setup script

This single call handles: **apt-install Python 3.11 + create the venv on
Drive**, ffmpeg install, light pip deps, CosyVoice clone, weights download,
CosyVoice pinned requirements (installed unmodified into the venv), and an
AutoModel import sanity check via the venv's python.

First-time run: ~10-15 minutes (mostly pip install of torch 2.3.1 and the
rest of CosyVoice's pinned deps). Warm re-runs: a few seconds.

```python
!python "{PROJ}/scripts/colab_setup.py"
```

---

## Cell 3 — Make the env vars visible to subsequent Colab cells

The setup script writes `scripts/colab_env.sh`; `source` it to inherit the
exported variables (Drive path, CosyVoice repo, model dir, venv path, Ollama
host) in this Colab Python process.

```python
!source "{PROJ}/scripts/colab_env.sh"
import os
for k in ("DRIVE_ROOT", "COSYVOICE_REPO_DIR", "COSYVOICE_MODEL_DIR",
          "COSYV_VENV_PYTHON", "OLLAMA_HOST"):
    print(f"  {k}={os.environ.get(k)}")
```

If `COSYV_VENV_PYTHON` is empty or points to a missing file, the venv
wasn't created properly — re-run Cell 2.

---

## Cell 4 — (Optional) Install or pull Ollama

Only run this if you want Ollama on the same Colab box. If you have Ollama
running elsewhere, set `OLLAMA_HOST` in `scripts/colab_env.sh` (or pass
`--ollama-host http://your-gpu-box:11434` to `colab_setup.py`) and skip
this cell.

```python
# Install Ollama
!curl -fsSL https://ollama.com/install.sh | sh
# Start the server in the background
import subprocess, time, urllib.request
serve = subprocess.Popen(["ollama", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(30):
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            if r.status == 200: break
    except Exception:
        time.sleep(1)
print("Ollama serving on http://localhost:11434")
# Pull the translation model
import ollama
ollama.pull(os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
!ollama list
```

> **Warning:** Running Ollama on the same T4 as CosyVoice3 risks OOM.
> Default to `OLLAMA_HOST=http://<your-gpu-box>:11434` and skip this cell.

---

## Cell 5 — Run the end-to-end pipeline test

```python
import os, sys
PROJ = os.environ["DRIVE_ROOT"]
# NOTE: we do NOT add COSYVOICE_REPO_DIR to sys.path in the main process
# anymore. CosyVoice now runs in a separate subprocess (the venv), so the
# main kernel never imports it. Only the venv's python sees CosyVoice.
for p in (PROJ,):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline

# EDIT THIS to a real, public, sub-60s Shorts/Reel before running.
TEST_URL = "https://www.youtube.com/shorts/REPLACE_ME"
TARGET_LANG = "Spanish"
OUTPUT_DIR = os.path.join(PROJ, "runs/test_run")

final = pipeline.run_pipeline(
    url=TEST_URL,
    target_lang=TARGET_LANG,
    output_dir=OUTPUT_DIR,
    ollama_model="llama3.1:8b",
    whisper_model_size="small",
)
print(f"\nFinal video: {final}")
from google.colab import files
files.download(final)
```

---

## Cell 6 — Sanity check: compare durations

```python
import os, subprocess, json
def dur(p):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", p], text=True)
    return float(json.loads(out)["format"]["duration"])

# Find the original and the dubbed video inside the run's work/ dir.
import glob
work_dir = os.path.join(OUTPUT_DIR, "work")
orig_candidates = glob.glob(f"{work_dir}/*.mp4")
orig = orig_candidates[0] if orig_candidates else None
dubbed = final
if orig:
    print(f"Original: {dur(orig):.2f}s")
    print(f"Dubbed:   {dur(dubbed):.2f}s")
    print(f"Delta:    {abs(dur(orig) - dur(dubbed)):.2f}s (should be < 0.5s)")
```

---

## Cleanup (optional)

```python
# Free GPU memory between long runs
import torch, gc
gc.collect(); torch.cuda.empty_cache()
# Optionally stop the CosyVoice subprocess if one is still running
import pipeline
pipeline.reset_models()
```

---

## Troubleshooting

- **"AutoModel import OK" never prints (Stage 8).** The venv's `pip install
  -r requirements.txt` (Stage 7) didn't complete, the Matcha-TTS submodule
  is missing, or Drive is out of space. Re-run Cell 2; check the stage
  output for which sub-step failed.
- **Stage 2 ("venv") failed: "Could not install Python 3.11 via apt."**
  Colab's base image may not have python3.11 in main repos, and the
  deadsnakes PPA add may have been blocked. Try
  `Runtime → Factory reset runtime` and re-run, or check your network.
- **`snapshot_download` raises 401 / RepositoryNotFoundError.** The HF
  repo id may have changed. Pass `--hf-model-id` to the setup script with
  a working id, and update `scripts/colab_setup.py` → `DEFAULT_HF_MODEL_ID`.
- **Ollama cell 4 hangs on `curl https://ollama.com/install.sh`.** You're
  probably on a restricted network. Skip local Ollama and set
  `OLLAMA_HOST` in `scripts/colab_env.sh` instead.
- **`pipeline.run_pipeline` raises `PipelineError("CosyVoice venv discovery
  file not found...")`.** Re-run Cell 2 to regenerate the discovery file.
- **CosyVoice subprocess keeps dying mid-run.** Check Colab's `/tmp` and
  the venv directory for disk space; GPU OOM leaves a hint in the
  subprocess's stderr (drained by `pipeline.py` and included in the error
  message).
- **Setup script did 5 GB of work but my fix wasn't picked up.** You
  probably forgot to re-run Cell 1 (`git pull`). Re-run cells in order
  1 → 2 → 3 → 5.

For anything that needs more than a one-line fix, append an entry to
`progress.md` → "### Errors & Fixes" so the next session has context.
