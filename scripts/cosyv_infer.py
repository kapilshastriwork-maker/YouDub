"""
YouDub — CosyVoice inference subprocess driver.

This script runs INSIDE the Python 3.11 venv that scripts/colab_setup.py
creates. The main Colab process (Python 3.13) spawns this as a long-running
subprocess and sends newline-delimited JSON jobs to it over stdin. Each job
asks for one segment of TTS to be synthesized; we run CosyVoice3's
inference_cross_lingual once and write the result back as a JSON line on
stdout. stderr is free for human-readable logging.

Protocol
--------
Input (stdin, one JSON object per line):
  {"text": "translated text",
   "ref_audio": "/abs/path/to/24k_mono_reference.wav",
   "out_path": "/abs/path/to/seg_0042.wav"}

  Special control message (no TTS is performed):
  {"_cmd": "shutdown"}

Output (stdout, one JSON object per line):
  {"ok": true,  "out_path": "...", "duration": 1.23}
  {"ok": false, "error": "repr of the exception"}

Errors are reported per-job so the main process can decide whether to abort
or continue. A non-recoverable error (e.g. model load failure) is reported
the same way; the main process should treat repeated errors as fatal.

The CosyVoice AutoModel is loaded once on the first job and cached. We
intentionally do NOT reload the model between jobs — that's the whole
point of keeping this subprocess alive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any, Optional


def _log(msg: str) -> None:
    """Log to stderr. stdout is the protocol channel — keep it clean."""
    print(f"[cosyv_infer] {msg}", file=sys.stderr, flush=True)


def _load_model(model_dir: str):
    """Load CosyVoice3 AutoModel from a model dir. Imports happen here so a
    import error is reported as a per-job failure rather than crashing
    the subprocess on startup.
    """
    import torchaudio  # noqa: F401  (silence unused import linter)
    from cosyvoice.cli.cosyvoice import AutoModel

    _log(f"loading AutoModel from {model_dir}")
    model = AutoModel(model_dir=model_dir)
    _log(f"AutoModel loaded; sample_rate={model.sample_rate}")
    return model


def _synth_one(model: Any, text: str, ref_audio: str, out_path: str) -> float:
    """Run one cross-lingual inference and save the result. Returns duration
    in seconds. Raises on any failure (caller turns the exception into a
    JSON error response).
    """
    import torchaudio  # type: ignore

    for chunk in model.inference_cross_lingual(
        tts_text=text, prompt_audio=ref_audio, stream=False
    ):
        wav = chunk["tts_speech"]  # torch.Tensor, shape (1, samples)
        torchaudio.save(out_path, wav.cpu(), model.sample_rate)
        samples = wav.shape[-1]
        return float(samples) / float(model.sample_rate)
    raise RuntimeError("inference_cross_lingual yielded no chunks")


def _handle_job(model: Any, job: dict) -> dict:
    """Validate a job dict and run one synthesis. Always returns a dict
    with at least an 'ok' key.
    """
    text = job.get("text", "")
    ref_audio = job.get("ref_audio", "")
    out_path = job.get("out_path", "")
    if not text:
        # No text — write a short silence file so the time-align stage
        # has a real file to work with. The main process does the same
        # thing in-process; we mirror the behaviour for consistency.
        duration = 0.1
        _write_silence_wav(out_path, duration)
        return {"ok": True, "out_path": out_path, "duration": duration, "silent": True}
    if not ref_audio or not os.path.isfile(ref_audio):
        return {"ok": False, "error": f"ref_audio not found: {ref_audio!r}"}
    if not out_path:
        return {"ok": False, "error": "out_path is required"}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    duration = _synth_one(model, text, ref_audio, out_path)
    return {"ok": True, "out_path": out_path, "duration": duration, "silent": False}


def _write_silence_wav(path: str, duration_sec: float) -> None:
    """Write a mono 24 kHz silent WAV of the given length. Mirrors
    pipeline._write_silence_wav but inlined so this script has no
    cross-package dependency on the main process.
    """
    import wave

    sr = 24000
    n = max(1, int(duration_sec * sr))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


def _main_loop(model: Any) -> int:
    """Read jobs from stdin, run them, write results to stdout. Returns
    the process exit code.
    """
    _log("entering main loop; waiting for jobs on stdin")
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"ok": False, "error": f"malformed JSON: {e}"})
            continue
        if not isinstance(job, dict):
            _send(
                {
                    "ok": False,
                    "error": f"job must be a JSON object, got {type(job).__name__}",
                }
            )
            continue
        # Control messages
        if job.get("_cmd") == "shutdown":
            _log("shutdown requested; exiting")
            return 0
        # Real job
        try:
            result = _handle_job(model, job)
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            _log(f"job failed: {e!r}\n{tb}")
            result = {"ok": False, "error": repr(e)}
        _send(result)
    _log("stdin closed; exiting")
    return 0


def _send(obj: dict) -> None:
    """Write one JSON response line to stdout and flush immediately so
    the main process can read it without buffering delays.
    """
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="YouDub CosyVoice inference driver (runs in the Py3.11 venv).",
    )
    parser.add_argument(
        "--model-dir", required=True, help="Path to the CosyVoice3 model directory."
    )
    parser.add_argument(
        "--ref-audio",
        required=True,
        help="Path to the 24 kHz mono reference clip for voice cloning.",
    )
    args = parser.parse_args(argv)

    # Make sure CosyVoice and Matcha-TTS are importable when this script
    # is invoked directly (the main process prepends them to sys.path for
    # its own imports, but this subprocess starts fresh).
    repo_dir = os.environ.get("COSYVOICE_REPO_DIR")
    if repo_dir:
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        matcha = os.path.join(repo_dir, "third_party", "Matcha-TTS")
        if matcha not in sys.path:
            sys.path.insert(0, matcha)

    # Load the model before entering the loop. If this fails, the main
    # process's stage_sanity_check will see it on the first stdout line.
    try:
        model = _load_model(args.model_dir)
    except Exception as e:
        tb = traceback.format_exc(limit=4)
        _log(f"AutoModel load failed: {e!r}\n{tb}")
        _send({"ok": False, "error": f"AutoModel load failed: {e!r}"})
        return 2

    return _main_loop(model)


if __name__ == "__main__":
    sys.exit(main())
