# YouDub — Colab Setup & Test Notebook

Copy each cell block into a fresh Colab notebook. The cells are idempotent
(rerunning them on a warm Drive won't re-download the world).

Recommended runtime: **T4 GPU**, **High RAM**.

---

## Cell 1 — Mount Drive and set paths

```python
from google.colab import drive
drive.mount('/content/drive')

# Where everything lives. Edit if you want a different Drive folder.
import os
DRIVE_ROOT = "/content/drive/MyDrive/YouDub"
os.makedirs(DRIVE_ROOT, exist_ok=True)
os.environ["DRIVE_ROOT"] = DRIVE_ROOT
os.environ["COSYVOICE_REPO_DIR"] = f"{DRIVE_ROOT}/CosyVoice"
os.environ["COSYVOICE_MODEL_DIR"] = f"{DRIVE_ROOT}/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
os.environ["OLLAMA_HOST"] = "http://localhost:11434"  # change if remote
!echo "DRIVE_ROOT=$DRIVE_ROOT"
```

---

## Cell 2 — System deps (ffmpeg, sox, git-lfs)

```python
!apt-get update -qq
!apt-get install -y -qq ffmpeg sox libsox-dev git-lfs
!git lfs install
!ffmpeg -version | head -1
```

---

## Cell 3 — Light Python deps (yt-dlp, faster-whisper, ollama, etc.)

```python
# These don't conflict with CosyVoice's pinned versions.
!pip install -q -U "yt-dlp>=2024.8.6" \
                   "faster-whisper>=1.0.3" \
                   "ffmpeg-python>=0.2.0" \
                   "huggingface_hub>=0.24.0" \
                   "ollama>=0.3.0" \
                   "requests>=2.32.0"
!pip install -q -U ipywidgets
```

---

## Cell 4 — Clone CosyVoice (idempotent)

```python
import os, subprocess
repo = os.environ["COSYVOICE_REPO_DIR"]
if not os.path.isdir(repo):
    print(f"Cloning CosyVoice into {repo} (one-time, takes a minute)...")
    os.makedirs(os.path.dirname(repo), exist_ok=True)
    subprocess.check_call([
        "git", "clone", "--recursive",
        "https://github.com/FunAudioLLM/CosyVoice.git", repo,
    ])
else:
    print(f"CosyVoice already at {repo}; pulling latest + updating submodules")
    subprocess.check_call(["git", "-C", repo, "pull", "--recurse-submodules"])
# Make sure submodules are present even on a partial previous clone.
subprocess.check_call(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
!ls "$COSYVOICE_REPO_DIR/third_party/Matcha-TTS" | head -3
```

---

## Cell 5 — Download CosyVoice3 weights from Hugging Face (idempotent)

```python
import os
from huggingface_hub import snapshot_download

model_dir = os.environ["COSYVOICE_MODEL_DIR"]
if os.path.isdir(model_dir) and os.path.isfile(f"{model_dir}/cosyvoice3.yaml"):
    print(f"Model already at {model_dir}; skipping download")
else:
    print(f"Snapshot-downloading FunAudioLLM/Fun-CosyVoice3-0.5B-2512 -> {model_dir}")
    snapshot_download(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=model_dir,
        # 9.75 GB total; allow resume
        max_workers=4,
    )
!du -sh "$COSYVOICE_MODEL_DIR"
```

---

## Cell 6 — Install CosyVoice's pinned requirements

```python
import os, subprocess
repo = os.environ["COSYVOICE_REPO_DIR"]
# Run pip from inside the clone so the relative requirements.txt path resolves.
print("Installing CosyVoice's pinned requirements (this can take 2-3 minutes)...")
subprocess.check_call(["pip", "install", "-q", "-r", "requirements.txt"], cwd=repo)
print("Done. Sanity-check:")
!python -c "import sys; sys.path.insert(0, '$COSYVOICE_REPO_DIR'); sys.path.insert(0, '$COSYVOICE_REPO_DIR/third_party/Matcha-TTS'); from cosyvoice.cli.cosyvoice import AutoModel; print('AutoModel import OK')"
```

> **If `import AutoModel` fails on Colab**, the most common cause is an
> onnxruntime / numpy version drift. Re-run with `pip install --upgrade
> onnxruntime numpy==1.26.4` in the CosyVoice directory.

---

## Cell 7 — Install Ollama (only if you want local translation)

```python
# Skip this cell entirely if OLLAMA_HOST points to a remote box.

!curl -fsSL https://ollama.com/install.sh | sh
# Run ollama serve in the background
import subprocess, time, os
os.makedirs(os.path.expanduser("~/.ollama"), exist_ok=True)
serve = subprocess.Popen(
    ["ollama", "serve"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
# Wait for the port to come up.
import urllib.request, socket
for _ in range(30):
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1) as r:
            if r.status == 200: break
    except Exception:
        time.sleep(1)
print("Ollama serving on http://localhost:11434")
```

> **Warning:** Ollama on the same T4 as CosyVoice3 will OOM. Default to
> `OLLAMA_HOST=http://<your-gpu-box>:11434` and skip this cell.

---

## Cell 8 — Pull the translation model

```python
import os, ollama
ollama.pull(os.environ.get("OLLAMA_MODEL", "llama3.1:8b"))
!ollama list
```

---

## Cell 9 — Add project to `sys.path` and smoke-test imports

```python
import os, sys
PROJ = "/content/drive/MyDrive/YouDub"  # the folder containing pipeline.py
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)
# Make CosyVoice importable too
if os.environ["COSYVOICE_REPO_DIR"] not in sys.path:
    sys.path.insert(0, os.environ["COSYVOICE_REPO_DIR"])
if os.path.join(os.environ["COSYVOICE_REPO_DIR"], "third_party", "Matcha-TTS") not in sys.path:
    sys.path.insert(0, os.path.join(os.environ["COSYVOICE_REPO_DIR"], "third_party", "Matcha-TTS"))

import pipeline
print("Pipeline version: Phase 1")
for name in ["download_video", "transcribe_audio", "translate_segments",
             "extract_reference_voice", "synthesize_dubbed_audio",
             "time_align_segment", "mux_final_video", "run_pipeline"]:
    print(f"  {name:30s} {getattr(pipeline, name).__doc__.splitlines()[0].strip()}")
```

---

## Cell 10 — Run the end-to-end test

```python
# EDIT THIS to a real, public, sub-60s Shorts/Reel before running.
TEST_URL = "https://www.youtube.com/shorts/REPLACE_ME"
TARGET_LANG = "Spanish"
OUTPUT_DIR = "/content/drive/MyDrive/YouDub/runs/test_run"

# Recommended first run: a 15-30s English clip -> Spanish, to validate timing.
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

## Cell 11 — Sanity check: compare durations

```python
# Run this AFTER the pipeline to confirm the dubbed video length matches the
# original within ~0.5s (it should, because we time-align each segment).
import subprocess, json
def dur(p):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", p], text=True
    )
    return float(json.loads(out)["format"]["duration"])

orig = "/content/drive/MyDrive/YouDub/runs/test_run/work/<id>.mp4"  # or copy
# final = OUTPUT_DIR + "/dubbed_Spanish.mp4"
# print(f"Original: {dur(orig):.2f}s   Dubbed: {dur(final):.2f}s")
```

---

## Cleanup (optional)

```python
# Free GPU memory between long runs
import torch, gc
gc.collect(); torch.cuda.empty_cache()
```
