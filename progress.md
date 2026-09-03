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
4. **yt-dlp may fail on Instagram without cookies.** YouTube may need
   PO tokens on `n`/`tv` clients. The setup cell does not export
   cookies; users hitting 401/403 will need to add `cookiefile` to the
   `dl_opts` dict in `download_video`.
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
│   └── colab_setup.py      # single source of truth for Colab setup
├── setup_colab.md          # 6-cell thin runner notebook guide
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

