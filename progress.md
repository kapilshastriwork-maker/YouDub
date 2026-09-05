# YouDub — Project Progress

AI dubbing pipeline that takes a short single-speaker video, transcribes,
translates, voice-clones, and produces a dubbed version. Built on a
Google Colab T4 GPU.

---

## Infrastructure

### Colab runner architecture (refactored)

**Problem.** Earlier versions of `setup_colab.md` contained 11 inline
notebook cells — apt installs, pip installs, CosyVoice cloning, weights
download, the AutoModel import test — written directly as cell code.
Every time something broke in Colab, the fix had to be re-described back
into the local repo, and the two drifted out of sync within a day.

**Fix (current).** The Colab notebook is now a thin runner. All setup
logic lives in `scripts/colab_setup.py` in this repo, and the notebook
contains only:

1. **Cell 1** — `git clone` / `git pull` the YouDub repo.
2. **Cell 2** — `!python scripts/colab_setup.py` (one call, all setup).
3. **Cell 3** — `!source scripts/colab_env.sh` so env vars cross the
   subshell boundary into the Colab Python process.
4. **Cell 4** — *(optional)* Install / pull Ollama if running locally.
5. **Cell 5** — Run the pipeline test (`pipeline.run_pipeline(...)`).
6. **Cell 6** — Sanity-check: original vs dubbed duration.

Any fix to the setup flow now happens once, in `scripts/colab_setup.py`,
and propagates to Colab on the next `git pull` + re-run.

**Why Python (not bash).** The setup needs `apt-get`, `pip`, `subprocess`,
HF `snapshot_download` (a Python call), and an `urllib` polling loop for
Ollama. A bash script would shell out to `python -c` for the Python parts
anyway. A single `.py` file is one Colab cell.

**Idempotency.** Each of the 7 stages in the script has a guard:
file-existence checks for the CosyVoice clone and weights download; a
`.youdub_cosyv_deps_installed` marker file for the heavy `pip install
-r requirements.txt`; `which` checks for system packages; `pip show`
checks for Python packages. Re-running the script is a no-op when the
environment is already warm.

**Environment variable propagation.** The script writes
`scripts/colab_env.sh` with the four key vars (`DRIVE_ROOT`,
`COSYVOICE_REPO_DIR`, `COSYVOICE_MODEL_DIR`, `OLLAMA_HOST`) and the HF
model id. A Colab cell then `source`s the file. `colab_env.sh` is in
`.gitignore` because it's host-specific (regenerated on every setup run).

**CLI flags.** The script accepts `--drive-root`, `--ollama-host`,
`--hf-model-id`, `--cosyvoice-branch`, and five `--skip-*` flags. The
99% case is no flags at all.

### Deprecation note

The 11-cell inline setup approach is **deprecated** as of this refactor.
Any older tutorial, doc, or comment that references inline cells is
outdated. The script is the single source of truth. If you find a gap,
add a stage to `colab_setup.py` rather than another inline cell.

---

## Python 3.13 compatibility

Colab's default kernel is now Python 3.13. CosyVoice3's exact pins
(`torch==2.3.1`, `numpy==1.26.4`, `onnxruntime-gpu==1.18.0`,
`openai-whisper==20231117`) have no cp313 wheels and several have no
sdist at all. We went through four approaches before landing on a clean
solution. The first three are documented here as historical record; the
fourth is the current architecture.

| # | Approach tried | Why it failed |
|---|---|---|
| 1 | Filter `onnxruntime-gpu` out of CosyVoice's `requirements.txt` and install a newer `>=1.20.0,<1.26.0` separately | Resolved the ORT pin, but `torch==2.3.1` (which has no wheel AND no sdist — PyTorch doesn't ship sdists) remained an unresolvable hard blocker. |
| 2 | Filter `openai-whisper` out too and install a newer release | `cosyvoice/cli/frontend.py:11` does `import whisper` and `frontend.py:98` calls `whisper.log_mel_spectrogram(speech, n_mels=128)` in the cross-lingual inference path. Not a "demo script" dep — actively used at inference time. The sanity check would fail immediately with `ModuleNotFoundError` if we filtered it. |
| 3 | Relax `torch==2.3.1` to a range with cp313 wheels (>=2.9.1) | CosyVoice3 is built against torch 2.3.1's specific Qwen2 LLM, SDPA, and tensor APIs. Bumping to 2.9.1+ risks silent model-quality regression (different numerics in attention, layernorm fp32 promotion, etc.). Untested territory. |
| 4 | Pin Colab to Python 3.12 via the runtime UI | Not viable. Colab removed the per-runtime Python version selector. Community kernel-swap hacks (e.g., manually downloading python.org tarballs) break Colab's own kernel bridge. |
| **5** | **Isolate CosyVoice in its own Python 3.11 venv on Drive** | **Chosen. The main Colab kernel (3.13) runs the pipeline orchestration; a long-running subprocess in the venv handles TTS over a JSON-over-stdio protocol. CosyVoice's `requirements.txt` is installed unmodified in the venv, where Py3.11 has full wheel support for all the pins. The 30-40 s model load is paid once and amortized across all segments.** |

### Architecture details

- **Main kernel (Py3.13)**: `pipeline.py` orchestration, faster-whisper
  (transcription), Ollama (translation), ffmpeg/yt-dlp (video IO), the
  `synthesize_dubbed_audio` function in pipeline.py spawns and manages
  the subprocess.
- **CosyVoice venv (Py3.11)**: lives at `$DRIVE_ROOT/cosyv_venv311` so
  it persists across Colab session reconnects. Created by
  `scripts/colab_setup.py` Stage 2 (apt-install python3.11 via main
  repos or deadsnakes PPA fallback, then `python3.11 -m venv`).
  CosyVoice's pinned `requirements.txt` is installed unmodified in
  Stage 7 via the venv's `pip`.
- **IPC protocol**: newline-delimited JSON over stdin/stdout. Main
  writes `{"text": "...", "ref_audio": "...", "out_path": "..."}\n`;
  venv writes `{"ok": true, "out_path": "...", "duration": 1.23}\n` (or
  `{"ok": false, "error": "..."}`). The venv-side driver is
  `scripts/cosyv_infer.py`; it loads `AutoModel` on the first job and
  holds it in memory for subsequent jobs.
- **Discovery**: `scripts/colab_setup.py` Stage 8 writes
  `$DRIVE_ROOT/cosyv_venv311/venv_python_path.json` containing
  `{"venv_dir", "venv_python", "model_dir", "cosyvoice_repo"}`.
  `pipeline.py._find_cosy_venv()` reads this file (searching
  `COSYV_VENV_DIR` env var first, then the default Drive location).
- **Crash recovery**: on subprocess death mid-run, `pipeline.py`
  respawns the subprocess once and retries the failed segment. If the
  respawn also fails, the segment raises `RuntimeError` with the
  drained stderr included in the message.

### Why this is better than the previous approaches

- **No filter/relax logic to maintain.** CosyVoice's `requirements.txt`
  is installed verbatim.
- **No risk of inference-path regression from a newer torch.** The
  venv runs the original torch 2.3.1.
- **Re-running the setup script is fast on warm sessions.** A marker
  file at `$DRIVE_ROOT/cosyv_venv311/.youdub_venv_ready` short-circuits
  Stages 2-7 in seconds; only the sanity check (Stage 8) re-runs the
  import test.
- **One model load per `run_pipeline` call, not per segment.** The
  long-running subprocess caches `AutoModel` across all segments.

### Tradeoffs accepted

- **~3 GB extra on Drive for the venv** (torch + onnxruntime-gpu + the
  full CosyVoice dep tree). The venv co-exists with the existing
  ~10 GB of CosyVoice weights on Drive; total Drive usage is ~15 GB.
- **No in-process AutoModel access.** Anything that previously called
  `pipeline._get_cosyvoice()` directly (only tests, in our case) now
  has to go through the subprocess protocol. This is a clean boundary
  but a real change for any future code that wants the model in-process.
- **Subprocess IPC has slightly higher per-job overhead than in-process
  calls** (~1-2 ms per JSON line vs. ~0 ms in-process). For 20-50
  segments this is invisible compared to the per-segment TTS latency
  (~1-3 s).

---

## Phase 1 — Core Pipeline

### What was built

All code lives in `pipeline.py` as eight public functions plus lazy model
singletons. Each function has a single responsibility and a documented
input/output contract, so the file can be imported by a future FastAPI
endpoint without restructuring.

| # | Function | Purpose | Returns |
|---|---|---|---|
| 1 | `download_video(url, output_dir, max_duration_sec=60)` | yt-dlp fetch + audio extract | `{video_path, audio_path, duration, title, uploader}` |
| 2 | `transcribe_audio(audio_path, model_size="small", device="cuda", compute_type="float16", language=None)` | faster-whisper transcription | `[{start, end, text}, ...]` |
| 3 | `translate_segments(segments, target_lang, ollama_model="llama3.1:8b")` | per-segment Ollama translation, with fallback | `[{start, end, original_text, text}, ...]` |
| 4 | `extract_reference_voice(audio_path, duration_sec=8, output_dir=None)` | middle-N-second clip → 24 kHz mono WAV | path to reference clip |
| 5 | `synthesize_dubbed_audio(segments, reference_voice_path, output_dir, model_dir=None)` | CosyVoice3 cross-lingual zero-shot per segment | `[{..., audio_path, synth_duration}, ...]` |
| 6 | `time_align_segment(audio_path, target_duration, output_path)` | ffmpeg atempo with chain for ratios outside [0.5, 2.0] | `output_path` |
| 7 | `mux_final_video(video_path, aligned_segments, output_path)` | ffmpeg filter_complex: adelay + concat + replace audio | `output_path` |
| 8 | `run_pipeline(url, target_lang, output_dir, ...)` | orchestrator with per-stage try/except | final video path |

Supporting infrastructure:
- `requirements.txt` — light deps (yt-dlp, faster-whisper, ollama, ffmpeg-python, huggingface_hub)
- `cosyv_requirements.txt` — readable reference of CosyVoice's pinned versions
- `setup_colab.md` — 11-cell Colab setup guide (mount Drive, clone repo, snapshot weights, install deps, smoke test, end-to-end run)
- `test_url.txt` — placeholder for the test Shorts/Reel URL

### Key technical decisions

- **faster-whisper over openai-whisper.** CTranslate2 backend is ~4× faster
  and uses far less VRAM at the same accuracy. `small` is the T4 default;
  `base` for very tight VRAM; `large-v3` for max quality.
- **faster-whisper `compute_type="float16"` on T4.** T4 has no BF16 tensor
  cores so `bfloat16` silently downcasts.
- **yt-dlp pre-probe for duration check.** `extract_info(url, download=False)`
  is a single network call, much cheaper than downloading then failing.
- **CosyVoice3 via `inference_cross_lingual`** (not `inference_zero_shot`).
  Cross-lingual is the canonical CosyVoice3 path for source-LANG audio →
  target-LANG speech and avoids needing a source-language `prompt_text`,
  which is the most common source of voice-clone quality regression.
- **CosyVoice3 weights via `snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512')`**
  into the cloned `FunAudioLLM/CosyVoice` repo's
  `pretrained_models/Fun-CosyVoice3-0.5B` subdirectory (the upstream-
  recommended layout; the local directory name drops the `-2512` suffix
  per the official README). The `cosyvoice` Python package and
  `Matcha-TTS` submodule come from the git clone, never from the HF
  repo.
- **Ollama host via `OLLAMA_HOST` env var, default `localhost:11434`.**
  Lets the same code work whether Ollama runs on the Colab T4 (will OOM
  with CosyVoice3 loaded), on the user's laptop, or on a separate GPU box.
- **Per-segment translation (not batched).** Preserves per-segment length
  budgeting, which is critical for downstream atempo ratios staying in a
  reasonable range.
- **Translation prompt enforces ±20% char-count budget.** Keeps synth
  durations close to originals so time-alignment ratios rarely need extreme
  atempo chaining.
- **Per-segment error recovery in `translate_segments`.** If Ollama returns
  empty/garbage, fall back to original text and log a warning. Never crash
  the pipeline on a single bad translation.
- **Atempo chaining for ratios outside [0.5, 2.0].** Math factored to
  produce a list of factors all in legal range. Heaviest change first,
  gentlest last (per ffmpeg docs — best audio quality).
- **Time-alignment via ffmpeg `atempo`, not rubberband.** Rubberband needs
  an extra system install; atempo is "good enough" for short TTS clips.
- **Final mux uses ffmpeg `filter_complex` with `adelay` + `concat`.** One
  invocation, video stream copied (`-c:v copy`), audio re-encoded to AAC.
  Gaps between segments are preserved as silence from the concat.
- **Audio replacement, not mixing.** Dubbed track replaces the original
  entirely. No background-music ducking (out of scope for Phase 1).
- **Lazy module-level singletons (`_get_whisper`, `_get_cosyvoice`,
  `_get_ollama`).** Model load is heavy; caching it once means the future
  FastAPI worker reuses the same model across requests. `reset_models()`
  provided for tests.
- **`run_pipeline` per-stage try/except** wraps each call so the error
  message identifies which stage failed. Raises `PipelineError` (a
  `RuntimeError` subclass) with the format `[stage: N/7 name] failed: ...`.

### Known limitations / TODOs

1. **Time-alignment quality not yet validated on real clips.** Pure atempo
   introduces small pitch/timbre drift; the ±20% translation length budget
   should keep ratios < 3:1 in practice, but unverified.
2. **Synthesized segment can exceed original gap+duration.** If the TTS
   clip is longer than the time slot, the next segment's start is delayed
   past its intended time. Mitigated by length prompt; not fixed.
3. **No segment overlap handling.** `mux_final_video` assumes the input
   segments' `start` values are non-decreasing and non-overlapping. Faster-
   whisper + VAD should produce this, but it isn't enforced.
4. **yt-dlp auth requires a cookies.txt file in Colab.** Set
   `COOKIEFILE_PATH` (or pass `cookiefile=` to `download_video` /
   `run_pipeline`) — see the "yt-dlp blocked by YouTube bot detection"
   entry in Errors & Fixes for setup details.
4a. **For demo-day risk mitigation, prefer `local_file_path` over `url` when YouTube bot detection is likely.** `load_local_video` bypasses yt-dlp entirely; see the "YouTube bot detection is intermittent" entry in Errors & Fixes. This is also the natural UX for a future file-upload feature in the FastAPI frontend.
5. **Ollama on the same Colab T4 will OOM once CosyVoice3 is loaded.**
   Combined VRAM ~3.5 GB; T4 has 16 GB so it should actually fit, but
   the first model load is slow and contention with the CUDA context is
   not measured.
6. **CosyVoice3 cross-lingual mode quality not yet validated on real
   en→es / en→hi clips.** Need to A/B test vs `inference_zero_shot` with
   a transcribed prompt.
7. **Reference voice clip is always the *middle* of the audio.** May
   capture silence or background music in some clips. A future improvement
   is to pick the highest-energy 8 s window.
8. **No retry logic for ffmpeg transient failures.** A single
   `CalledProcessError` bubbles up.
9. **No streaming for long Ollama responses.** We use the non-streaming
   `chat()` API, which is fine for short segments.
10. **No background-music preservation.** The `mix_with_original` parameter
    promised in the plan was not implemented; the spec is "replace audio
    entirely" so this is intentionally out of scope.

### File layout

```
YouDub/
├── .gitignore              # excludes weights, generated media, secrets, colab_env.sh
├── pipeline.py             # the 8 functions + helpers (this is the core)
├── requirements.txt        # light deps (install first)
├── cosyv_requirements.txt  # reference copy of CosyVoice's pinned deps
├── scripts/
│   ├── colab_setup.py      # single source of truth for Colab setup
│   └── cosyv_infer.py      # long-running CosyVoice subprocess driver (runs in venv)
├── setup_colab.md          # 5-cell thin runner notebook guide
├── test_url.txt            # placeholder for the test Shorts URL
└── progress.md             # this file
```

### Repository hygiene

A `.gitignore` was added to keep the repo small and reviewable during the
hackathon. Excludes:

- **Python artifacts** (`__pycache__/`, `*.pyc`, `.ruff_cache/`, `.venv/`,
  `*.egg-info/`) — never useful in source control.
- **Generated media** (`outputs/`, `*.mp4`, `*.wav`, `*.mp3`, plus other
  common audio/video extensions) — these are pipeline outputs, large,
  and fully regenerable from `pipeline.py`.
- **Model weights and large downloads** (`*.pt`, `*.onnx`, `*.safetensors`,
  `pretrained_models/`, `CosyVoice/`) — 9.75 GB for CosyVoice3 alone.
  These belong on Google Drive, not in git.
- **Secrets and local config** (`.env`, `*.env`, `ngrok.yml`) — the ngrok
  authtoken and any future API keys stay local.
- **OS/editor junk** (`.DS_Store`, `Thumbs.db`, `.vscode/` with
  `settings.json` and `extensions.json` allow-listed) — `.vscode/settings.json`
  is kept so shared editor config (e.g., Python interpreter, ruff rules)
  can be reviewed and reproduced.
- **Jupyter checkpoints** (`.ipynb_checkpoints/`) — but `.ipynb` files
  themselves are committed with outputs intact, as evidence the pipeline
  works end-to-end.

Net effect: `git status` after a fresh run should show only the source
files plus the output-laden notebooks, never the 10 GB of weights.

### Future phases (not started)

- **Phase 2 — FastAPI wrapper + ngrok.** Wrap each function in an endpoint.
  Use the same `_get_*` singletons so the heavy models load once at app
  startup.
- **Phase 3 — Quality improvements.** ASR with word-level timestamps for
  tighter alignment; reference-voice energy windowing; background-music
  ducking; multi-speaker diarization.
- **Phase 4 — Production hardening.** Job queue, retry/backoff, S3 for
  outputs, observability.

---

### Errors & Fixes

*(Format:*

- **`<date>` — `<short error message>`**
  - What happened:
  - What I tried:
  - What worked:
  - Root cause (if known):
  - Fix to apply in code:
)*

- **mid-hackathon — `snapshot_download('agiws/Fun-CosyVoice3-0.5B')` raised `RepositoryNotFoundError` (401)**
  - What happened: First end-to-end Colab run died at the weights-download
    cell with a HF 401 on the agiws mirror; the repo had gone private
    or been taken down.
  - What I tried: Re-running the cell, refreshing the HF token — both failed
    because the namespace itself was returning 401.
  - What worked: Switching to the official upstream weights repo
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`, which is public and mirrors
    the same `cosyvoice3.yaml` + `.pt`/`.onnx` file layout.
  - Root cause: The agiws namespace was a third-party mirror; the owner
    removed/restricted it. We had been using it as a convenience and should
    have used the canonical FunAudioLLM repo from the start.
  - Fix applied: `pipeline.py` — `HF_MODEL_ID` env var now defaults to
    `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`. `setup_colab.md` Cell 5 now
    `snapshot_download`s from the same id into
    `pretrained_models/Fun-CosyVoice3-0.5B` (no `-2512` suffix in the
    local dir, matching the upstream README's example). `progress.md`
    decision log updated. Bonus simplification: the official repo contains
    only weights, which is actually a better fit for our existing setup —
    the Python source already comes from the `FunAudioLLM/CosyVoice` git
    clone, so we never needed the agiws bundled `cosyvoice_src/` folder.

- **post-Phase-1 — Colab ↔ local repo drift from inline setup cells**
  - What happened: Earlier versions of `setup_colab.md` had all setup
    logic (apt installs, pip, clone, weights, sanity check) baked into
    11 inline Colab cells. Every Colab debugging cycle required
    re-describing the fix into the local repo, and the two drifted
    within a session. The agiws-401 incident (previous entry) was
    harder to fix than it needed to be because the swap had to be made
    in both places.
  - What I tried: Adding more inline cells to handle edge cases —
    rapidly made the notebook unreadable.
  - What worked: Refactoring the notebook to a 6-cell thin runner and
    moving all setup into `scripts/colab_setup.py`. See
    `## Infrastructure` section above for the full design.
  - Root cause: Setup was treated as notebook content rather than repo
    content. Repos are version-controlled and shared; notebooks are
    session-local. The split was wrong.
  - Fix applied: New `scripts/colab_setup.py` (7 idempotent stages,
    CLI flags, writes `colab_env.sh` for env-var propagation). New
    `## Infrastructure` section in this file documents the architecture.
    `setup_colab.md` reduced from 228 lines (11 cells) to ~140 lines
    (6 cells + troubleshooting). The old inline-cell approach is
    deprecated — any future setup changes go into the script, not
    the notebook.

- **post-Phase-1 — `onnxruntime-gpu==1.18.0` not installable on Colab (no matching distribution)**
  - What happened: `pip install -r CosyVoice/requirements.txt` failed with
    `Could not find a version that satisfies the requirement
    onnxruntime-gpu==1.18.0`. CosyVoice's pinned version is from May 2024
    and has no cp313 wheel; Colab now defaults to Python 3.13.
  - What I tried: First instinct was to edit `CosyVoice/requirements.txt`
    in the cloned repo. Realised that would (a) be wiped on the next
    `git pull` and (b) leave no record of the override in our own code.
  - What worked: Two-step install in `scripts/colab_setup.py`:
    1. Filter the upstream `requirements.txt` content into a temp file
       with onnxruntime* lines removed, then `pip install -r` the
       filtered copy (gets everything else from the pin list normally,
       just skips the two offending packages). See the next entry for
       why we *don't* use `pip install --exclude`.
    2. `pip install "onnxruntime-gpu>=1.20.0,<1.26.0"` (Linux) or
       `"onnxruntime>=1.20.0,<1.26.0"` (Win/mac — auto-selected by
       `sys.platform`).
  - Root cause: Microsoft ships `onnxruntime-gpu` cp313 wheels starting at
    1.20.0 (Nov 2024). CosyVoice's 1.18.0 pin predates Python 3.13 and
    was never updated. CosyVoice's own onnxruntime API surface is
    vanilla (only `SessionOptions`, `GraphOptimizationLevel.ORT_ENABLE_ALL`,
    `InferenceSession`, and `session.run`), so 1.20+ is API-compatible
    with no functional difference.
  - Fix applied: New `ONNXRUNTIME_VERSION = ">=1.20.0,<1.26.0"`
    constant at the top of `scripts/colab_setup.py`. New
    `_onnxruntime_pkg_name()` helper picks `onnxruntime-gpu` (Linux)
    or `onnxruntime` (Win/mac) to mirror CosyVoice's own
    `sys_platform` markers. `stage_install_cosyv_deps` now does the
    two-step install and reports the actual installed version via
    `pip show` so the log shows what landed. Existing marker-file
    idempotency (`.youdub_cosyv_deps_installed`) covers both passes.
    No edit to upstream `CosyVoice/requirements.txt` (would be wiped
    on re-clone) and no edit to our reference `cosyv_requirements.txt`
    (it's a faithful copy of upstream; the docstring in the script
    explains where the override happens). Bump `ONNXRUNTIME_VERSION`
    in the script if a future CosyVoice release changes its onnx
    requirements.

- **post-Phase-1 — `pip install --exclude` is not a valid flag (usage error, exit 2)**
  - What happened: First attempted fix for the onnxruntime pin problem
    (previous entry) used `pip install -r requirements.txt --exclude
    onnxruntime --exclude onnxruntime-gpu`. That command failed
    immediately with `no such option: --exclude` and exit code 2.
  - What I tried: Looked up the pip docs to see if I'd gotten the
    flag name wrong. Found that `--exclude` is only valid on
    `pip download` and `pip list`, never on `pip install`. It's a
    common mistake — the option exists in pip's option parser, just
    not in the `install` command.
  - What worked: Filter the requirements file content into a temp file,
    then `pip install -r` the filtered copy. The filter helper
    `_filter_requirements_txt` in `scripts/colab_setup.py` drops any
    line whose first non-whitespace token starts with `onnxruntime`
    (covers `onnxruntime==1.18.0` and `onnxruntime-gpu==1.18.0` with
    or without `; sys_platform == ...` markers). It preserves
    `--extra-index-url` lines, blank lines, comments, and `onnx==1.16.0`
    (the ONNX format library, a different package). The filtered
    content is written to `tempfile.gettempdir() /
    youdub_cosyv_requirements_filtered.txt` and pip-installed from
    there. On success the file is cleaned up; on failure it's left in
    place for post-mortem inspection.
  - Root cause: I assumed pip's `--exclude` flag was universally
    supported because it's the obvious mechanism for "skip these
    packages". It's not. `pip install` resolves requirements verbatim
    — if you want to skip a line, you have to either edit the file or
    filter its content.
  - Fix applied: New `_filter_requirements_txt()` helper and
    `FILTERED_REQ_FILENAME` constant in `scripts/colab_setup.py`.
    `stage_install_cosyv_deps` now: reads the upstream requirements,
    filters onnxruntime* lines, writes the filtered copy to a temp
    file, installs from that, then runs the separate onnxruntime
    install pass as before. Added a comment in the script warning
    future maintainers not to re-introduce `--exclude`.

- **post-Phase-1 — `_run()` swallowed subprocess failure output (recurring pain)**
  - What happened: Every time a subprocess (pip install, apt-get, git,
    etc.) failed inside `scripts/colab_setup.py`, the script printed
    `subprocess.CalledProcessError: Command '...' returned non-zero exit
    status N` and nothing else. To see the actual error, we had to
    manually re-run the failing command outside the script. This
    happened at least three times during the hackathon — once for the
    onnxruntime pin, once for a transient HF download error, and once
    for a stale git lockfile.
  - What I tried: Reading `CalledProcessError.__str__` to see if there
    was a hidden kwarg. There isn't — Python's stdlib deliberately
    doesn't include captured output in the exception's string form, so
    `raise` from inside `subprocess.run` leaves us no chance to print
    anything.
  - What worked: Wrapped the `subprocess.run` call inside `_run()` in
    a `try/except CalledProcessError`. On failure, print a small header
    plus the last 3000 characters of the captured output (which already
    includes stderr because `_run` uses `stderr=subprocess.STDOUT`),
    then re-raise. The header includes a "skipped N" line if the output
    was truncated. A `UnicodeEncodeError` guard around the print means
    a single weird byte in pip's output can't crash the script just to
    report on a different error. Refactored `stage_sanity_check` to
    also go through `_run(check=False)` so the AutoModel import sanity
    check gets the same treatment.
  - Root cause: `subprocess.run(check=True)` raises inside the call
    before we get a chance to inspect the `CompletedProcess`, and
    `CalledProcessError` doesn't carry output in its `__str__`. We
    needed to either: (a) set `check=False` everywhere and handle
    return codes ourselves, or (b) wrap each call in a try/except.
    Option (b) keeps the API the same for the 9 existing call sites
    (they still get free `check=True` semantics) and is a smaller
    diff.
    - Fix applied: New `_print_failure_output(pretty_cmd, output)` helper
      in `scripts/colab_setup.py` that handles the truncation, header,
      and UnicodeEncodeError guard. `_run()` now wraps the
      `subprocess.run` call in a try/except that calls the helper and
      re-raises. New `_FAILURE_OUTPUT_TAIL_CHARS = 3000` constant
      controls the truncation. `check=False` call sites (currently just
      the `git pull` non-fatal-failure path) are unaffected — they
      don't raise, so the helper isn't called. Success-path output is
      still silent. From now on, a failing pip install inside the script
      will print the real error message directly in the script's output,
      no manual re-run required.

- **post-Phase-1 — Architecture: isolate CosyVoice in a Python 3.11 venv**
  - What happened: After filtering out `onnxruntime-gpu` and
    `openai-whisper` from CosyVoice's `requirements.txt` and installing
    newer compatible versions, the next blocker was `torch==2.3.1` —
    CosyVoice's exact pin has no cp313 wheel AND no sdist (PyTorch
    doesn't ship sdists). Considered relaxing the pin to a cp313 range
    (>=2.9.1), but that risks Qwen2 LLM numerics drift in CosyVoice3's
    inference path. Considered pinning Colab to Python 3.12 via the
    runtime UI, but that option was removed by Google. Considered
    community kernel-swap hacks, but they break Colab's own kernel
    bridge. The clean answer: don't try to make CosyVoice run in the
    Colab kernel at all. Isolate it.
  - What I tried: All four of the above, in sequence. The first two
    taught us the boundary of what we could fix in-process. The third
    was a real risk we weren't willing to take. The fourth wasn't
    available. See the "## Python 3.13 compatibility" section above
    for the full table.
  - What worked: `scripts/colab_setup.py` gets a new Stage 2 that
    apt-installs Python 3.11 (trying main repos first, falling back to
    the deadsnakes PPA, with a clear failure message if both fail) and
    creates `$DRIVE_ROOT/cosyv_venv311`. Stage 7 installs CosyVoice's
    ORIGINAL, UNMODIFIED `requirements.txt` into the venv — no
    filtering, no pin-relaxing. Stage 8 runs the AutoModel sanity
    check via the venv's python and writes
    `venv_python_path.json` so the main process can discover the
    interpreter. `pipeline.py`'s `synthesize_dubbed_audio` is refactored
    to spawn `scripts/cosyv_infer.py` as a long-running subprocess and
    talk to it over a newline-delimited JSON protocol on stdin/stdout.
    The subprocess loads `AutoModel` once and holds it across all
    segments, so the 30-40 s model load is paid once per
    `run_pipeline` call, not per segment.
  - Root cause: Colab's default kernel is Python 3.13, and CosyVoice3
    is built against a specific Py3.11 ecosystem. The two are
    fundamentally incompatible for in-process CosyVoice execution. The
    venv is the cleanest available boundary.
  - Fix applied: New `scripts/cosyv_infer.py` (long-running CosyVoice
    driver, ~150 lines). Major refactor of
    `scripts/colab_setup.py` (added Stage 2 venv creation, reverted the
    filter helpers from the previous two turns, added a new env var
    `COSYV_VENV_PYTHON` to `colab_env.sh`, added `--skip-venv` flag).
    Major refactor of `pipeline.py` (replaced in-process
    `_get_cosyvoice` with subprocess-based `_find_cosy_venv` +
    `_start_cosyv_subprocess` + `_send_cosyv_job` +
    `_stop_cosyv_subprocess` helpers; deleted `_get_cosyvoice`,
    `_cosyvoice_synth_one`, `_add_cosyvoice_to_path`, the in-process
    `import torchaudio` in `synthesize_dubbed_audio`, and the
    `_COSYVOICE_MODEL` / `_COSYVOICE_MODEL_DIR` module-level cache).
    `setup_colab.md` rewritten with a prominent
    "Two-environment architecture" section. `progress.md` gets a new
    "## Python 3.13 compatibility" section that documents the four
    failed approaches and the one that worked. The previous
    onnxruntime-filter and openai-whisper-filter entries stay as
    historical record of the path we took.

- **post-Phase-1 — `openai-whisper==20231117` build failed: `ModuleNotFoundError: No module named 'pkg_resources'`**
  - What happened: Stage 7's `pip install -r requirements.txt` inside the
    Py3.11 venv failed while building openai-whisper. Verbose pip log
    showed the import error from openai-whisper's old setup.py.
  - What I tried: Confirmed it's NOT a Python 3.13 issue (we're in the
    Py3.11 venv now). Confirmed openai-whisper==20231117 has no upper
    pin conflict. The earlier `## Python 3.13 compatibility` investigation
    had filtered openai-whisper out and installed a newer release, but
    that was solving the wrong problem — this is a setuptools regression.
  - What worked (part 1): Pinning `setuptools<80` into the venv *before*
    installing CosyVoice's requirements.txt. setuptools 80 split
    pkg_resources out of the default install into a separate
    `setuptools-pkg-resources` package, so venvs created with newer
    setuptools no longer have `pkg_resources` importable. openai-whisper
    20231117 imports it directly at build time. Also dropped the `-U`
    flag from the requirements.txt install — CosyVoice's pins are
    intentional and we don't want pip to silently upgrade past a
    version the Qwen2 LLM path was built against.
  - **Part 1 was insufficient on its own** — see the next entry for
    the build-isolation follow-up.
  - Root cause (part 1): setuptools 80 split pkg_resources out of the
    default install; openai-whisper 20231117's setup.py imports it
    directly. Unrelated to the Python 3.13 / CosyVoice-pin story.
  - Fix applied (part 1): `stage_install_cosyv_deps` now bootstraps
    `setuptools<80` into the venv first, then installs
    `requirements.txt` without `-U`. Marker file semantics unchanged
    (still guards the whole stage). The setuptools install runs on
    every invocation — no separate marker, since it's cheap (~2s) and
    the version pin is idempotent. **This entry is kept as historical
    record; the actually-correct fix is the build-isolation entry
    below.**

- **post-Phase-1 — `setuptools<80` alone didn't fix openai-whisper's build (pip build isolation)**
  - What happened: After installing `setuptools<80` into the venv,
    openai-whisper's build still failed with the same
    `ModuleNotFoundError: No module named 'pkg_resources'`. Direct
    check inside the venv confirmed `setuptools` 79.x and
    `pkg_resources` were both importable.
  - What I tried: Re-running Stage 7, re-checking the venv's
    `setuptools` version (correct, 79.x). Suspected a different cause
    when the same error recurred.
  - What worked: pip's default build isolation creates a fresh,
    throwaway environment for each package's build step, seeded with
    the latest setuptools from PyPI — completely ignoring the venv's
    installed packages. So `setuptools<80` in the venv is invisible
    to the build subprocess. Fix: install openai-whisper separately
    with `--no-build-isolation`, which tells pip to use the venv's
    own environment for build deps. Two preconditions must be in
    place in the venv first: (1) `setuptools<80` (so pkg_resources
    is bundled), (2) `wheel` (so bdist_wheel works without an
    isolated env). openai-whisper is also filtered out of the main
    `requirements.txt` install (via a temp-file filter, same pattern
    as the historical onnx filter) so the main batch doesn't fail
    trying to build it with the default isolated env.
  - Root cause: pip's build-isolation feature fetches its own
    setuptools into a throwaway build env per package, so venv-level
    setuptools pins are not visible to the build step.
    `--no-build-isolation` opts out of this for the specific package.
  - Fix applied: `stage_install_cosyv_deps` now (1) installs
    `setuptools<80 wheel` into the venv (combined call), (2) runs a
    filtered `requirements.txt` install with `openai-whisper` removed
    (via the new `_filter_requirements_txt(path, drop_pkgs)` helper,
    which writes the filtered copy to a temp file — deleted on
    success, left in place for post-mortem on failure), (3) runs a
    separate `pip install --no-build-isolation openai-whisper==20231117`
    so it builds against the venv's setuptools<80. Marker file
    semantics unchanged. The setuptools+wheel install runs on every
    invocation (no separate marker) — cheap (~2-3s) and the version
    pin is idempotent.

- **post-Phase-1 — yt-dlp blocked by YouTube bot detection on Colab ("Sign in to confirm you're not a bot")**
  - What happened: `download_video` failed on Colab with YouTube's
    bot-detection interstitial. Google Cloud IPs are flagged more
    aggressively than residential IPs, and yt-dlp without
    authentication can't bypass the check.
  - What I tried: Re-running, retry-with-backoff — same result. The
    underlying IP is the problem, not a transient failure.
  - What worked: yt-dlp accepts a `cookiefile` (Netscape-format
    `cookies.txt`) that authenticates as a logged-in browser session.
    Export from a browser extension like "Get cookies.txt LOCALLY"
    (Chrome/Firefox), upload to Drive or your local working dir, and
    supply it via either the `COOKIEFILE_PATH` env var (one-time,
    picked up everywhere) or the `cookiefile=` parameter directly.
  - Root cause: YouTube's per-IP bot detection on cloud-provider
    ranges. Cookies authenticate as a real browser session.
  - Fix applied: `download_video` and `run_pipeline` now accept an
    optional `cookiefile` parameter (default `None`, fully backward
    compatible). Resolution order: explicit `cookiefile` arg →
    `COOKIEFILE_PATH` env var → `None`. If set but the file is
    missing, `download_video` raises `RuntimeError` early (before any
    network call) with a remediation message. The cookiefile is
    passed to BOTH the metadata probe and the actual download — the
    probe can also trigger the bot check. The CLI smoke test got a
    new `--cookiefile` flag. Cookies are NOT committed to the repo
    (they're secrets) — users keep them on Drive or locally. The
    Known Limitations #4 entry was updated to point at the new
    functionality (the old text was factually wrong after this fix;
    it told users to edit `dl_opts` manually, which is no longer
    needed).

- **post-Phase-1 — YouTube bot detection is intermittent on Colab; added local-file fallback for demo safety**
  - What happened: The `cookiefile` fix (previous entry) made most
    downloads work, but YouTube's bot detection is heuristic and
    cloud-IP-restricted — a non-zero fraction of attempts still fail
    even with valid cookies. For a live demo recording, this is a
    real risk: a single failed download can derail the whole
    presentation.
  - What I tried: Retries, longer sleeps, multiple cookie files —
    all reduce but don't eliminate the failure rate. The underlying
    signal (cloud IP, heuristic check) is not deterministic.
  - What worked: Bypass yt-dlp entirely. Add a direct local-file
    input path. The user uploads a video via Colab's
    `google.colab.files.upload()` (or any other means) and we treat
    it as if `download_video` had just produced it. Same downstream
    interface, zero changes to stages 2-7. The URL path remains
    primary; local file is the documented fallback.
  - Root cause: YouTube's bot detection is heuristic and
    cloud-IP-restricted, not deterministic. Any in-band fix (cookies,
    retries, throttling) has a non-zero failure rate. An out-of-band
    path is more reliable than chasing the in-band failure rate
    to zero.
  - Fix applied: New `load_local_video(file_path, output_dir,
    max_duration_sec=60) -> dict` function in `pipeline.py` with
    the same return shape as `download_video`. Reuses
    `_ffmpeg_extract_audio` and `_ffprobe_duration` so the audio
    path is bit-identical to the URL flow. `run_pipeline` got a
    new optional `local_file_path` parameter — if provided, used as
    Stage 1 input (bypassing yt-dlp); if not, falls back to
    `download_video(url=...)`. Silent precedence: if both `url` and
    `local_file_path` are passed, `local_file_path` wins. The URL
    is still required by the signature for backward compatibility
    but can be a dummy value like `"local"` when
    `local_file_path` is used. CLI smoke test got a new
    `--local-file` flag. Known Limitations #4a was added to point
    at the new fallback. This is also a natural direct feature for
    the future FastAPI frontend (Phase 2): users can upload a file
    instead of pasting a URL, which is both a better UX and removes
    the bot-detection risk entirely from the user-facing flow.

- **post-Phase-1 — faster-whisper CUDA runtime mismatch on Colab's T4 driver**
  - What happened: `transcribe_audio` failed on Colab with
    "CUDA driver version is insufficient for CUDA runtime version".
    The T4 hardware is fine; faster-whisper's bundled CUDA runtime
    (via CTranslate2) requires a newer NVIDIA driver than the Colab
    T4 image ships by default.
  - What I tried: Re-installing the CUDA toolkit, pinning ctranslate2
    to an older build — same result. The driver/runtime version pair
    on the Colab image is the boundary; the bundled runtime is too
    new for it across multiple ctranslate2 versions.
  - What worked: Switch the default to CPU with int8 quantization.
    For 30-50s clips, CPU transcription is fast enough (a few seconds
    on a Colab host) and keeps the GPU fully available for CosyVoice3
    with no contention. GPU remains opt-in via `device="cuda"` for
    hosts with a correct driver/runtime pair.
  - Root cause: Colab's T4 image driver is too old for faster-whisper's
    bundled CUDA runtime. Either pin the runtime back (fragile, version-
    sensitive) or stop using GPU for this stage. CPU is the right call:
    transcription is lightweight relative to TTS, and we want the GPU
    dedicated to CosyVoice3.
  - Fix applied: `transcribe_audio` defaults changed to
    `device="cpu"`, `compute_type="int8"`. `run_pipeline` defaults
    `whisper_device="cpu"`, `whisper_compute_type="int8"`. CLI smoke
    test got `--whisper-device` and `--whisper-compute-type` flags so
    GPU can still be forced on a working host. The `_get_whisper` cache
    already invalidates correctly on param changes, so callers that
    switch mid-session are handled. No change to `setup_colab.md` (the
    new defaults are picked up automatically by the existing
    `run_pipeline` call).

