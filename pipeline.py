"""
YouDub — AI video dubbing pipeline (Phase 1).

Each function has a single responsibility and a clear input/output contract so
the same code can be (a) called from a Colab notebook, (b) unit-tested in
isolation, and (c) wrapped in a FastAPI endpoint later.

Module-level model objects (Whisper, CosyVoice, Ollama client) are lazily
constructed via the `_get_*` helpers and cached. This keeps the public
functions cheap to call and lets a FastAPI worker reuse the heavy models
across requests.

Conventions
-----------
- Every stage returns plain Python data (str / list[dict] / dict) so it can
  be JSON-serialised for an API response.
- Every stage that can fail does so with a typed exception (ValueError,
  RuntimeError, subprocess.CalledProcessError) plus a message that includes
  the stage name. `run_pipeline` catches these and re-raises as
  PipelineError after logging.
- All file paths are absolute strings; callers pass in `output_dir` and we
  never assume the cwd.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

logging.basicConfig(
    level=os.environ.get("YOUDUB_LOG", "INFO"),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("youdub")


# ----------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------


class PipelineError(RuntimeError):
    """Raised by `run_pipeline` when a stage fails. The message names the stage."""


# ----------------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------------

# Where the FunAudioLLM/CosyVoice repo is cloned. The setup cell places this
# under Google Drive so the weights persist across Colab sessions.
COSYVOICE_REPO_DIR = os.environ.get(
    "COSYVOICE_REPO_DIR",
    "/content/drive/MyDrive/YouDub/CosyVoice",
)

# Local path the snapshot_download call writes weights into. Must live inside
# the CosyVoice repo so AutoModel's relative yaml paths resolve correctly.
COSYVOICE_MODEL_DIR = os.environ.get(
    "COSYVOICE_MODEL_DIR",
    f"{COSYVOICE_REPO_DIR}/pretrained_models/Fun-CosyVoice3-0.5B",
)

# Ollama host. Defaults to localhost; set OLLAMA_HOST=http://gpu-box.lan:11434
# if Ollama runs on a separate machine.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Hugging Face model id. The `-2512` suffix is the official 2025-12 release of
# Fun-CosyVoice3-0.5B (base weights only — no bundled Python source). The
# `cosyvoice` Python package comes from the FunAudioLLM/CosyVoice Git clone,
# configured via COSYVOICE_REPO_DIR. The previously used agiws/Fun-CosyVoice3-0.5B
# mirror became unreachable (401) during the hackathon.
HF_MODEL_ID = os.environ.get(
    "YOUDUB_HF_MODEL_ID", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
)


# ----------------------------------------------------------------------------
# Lazy model singletons
# ----------------------------------------------------------------------------
# Pattern: each `_get_X()` returns the cached object, building it on first
# call. The `reset_models()` helper is provided for tests that need a clean
# state.


def _add_cosyvoice_to_path() -> str:
    """Put the CosyVoice repo + Matcha-TTS submodule on sys.path.
    Idempotent. Returns the cosyvoice repo dir (handy for error messages).
    """
    repo = Path(COSYVOICE_REPO_DIR)
    if not repo.is_dir():
        raise PipelineError(
            f"CosyVoice repo not found at {repo}. Run the Colab setup cell "
            f"that does `git clone --recursive https://github.com/"
            f"FunAudioLLM/CosyVoice.git {repo}` first."
        )
    matcha = repo / "third_party" / "Matcha-TTS"
    for p in (str(repo), str(matcha)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return str(repo)


_WHISPER_MODEL = None
_WHISPER_MODEL_SIZE = None
_WHISPER_DEVICE = None
_WHISPER_COMPUTE_TYPE = None


def _get_whisper(model_size: str, device: str, compute_type: str):
    """Return a cached `faster_whisper.WhisperModel`.
    Recreates the object if any of the construction params changed."""
    global _WHISPER_MODEL, _WHISPER_MODEL_SIZE, _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE
    if (
        _WHISPER_MODEL is not None
        and _WHISPER_MODEL_SIZE == model_size
        and _WHISPER_DEVICE == device
        and _WHISPER_COMPUTE_TYPE == compute_type
    ):
        return _WHISPER_MODEL
    from faster_whisper import WhisperModel  # heavy import, deferred

    log.info(
        "Loading faster-whisper model=%s device=%s compute=%s",
        model_size,
        device,
        compute_type,
    )
    _WHISPER_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
    _WHISPER_MODEL_SIZE = model_size
    _WHISPER_DEVICE = device
    _WHISPER_COMPUTE_TYPE = compute_type
    return _WHISPER_MODEL


_COSYVOICE_MODEL = None
_COSYVOICE_MODEL_DIR = None


def _get_cosyvoice(model_dir: str):
    """Return a cached `cosyvoice.cli.cosyvoice.AutoModel`. Slow on first call
    (~30-40 s for CosyVoice3-0.5B) so we cache aggressively."""
    global _COSYVOICE_MODEL, _COSYVOICE_MODEL_DIR
    if _COSYVOICE_MODEL is not None and _COSYVOICE_MODEL_DIR == model_dir:
        return _COSYVOICE_MODEL
    _add_cosyvoice_to_path()
    from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore  # noqa: E402

    log.info("Loading CosyVoice model from %s (first call is slow)...", model_dir)
    _COSYVOICE_MODEL = AutoModel(model_dir=model_dir)
    _COSYVOICE_MODEL_DIR = model_dir
    log.info("CosyVoice loaded. sample_rate=%s", _COSYVOICE_MODEL.sample_rate)
    return _COSYVOICE_MODEL


def _get_ollama():
    """Return an `ollama.Client` bound to OLLAMA_HOST."""
    import ollama  # type: ignore

    return ollama.Client(host=OLLAMA_HOST)


def reset_models() -> None:
    """Drop all cached model objects. Used by tests."""
    global _WHISPER_MODEL, _WHISPER_MODEL_SIZE, _WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE
    global _COSYVOICE_MODEL, _COSYVOICE_MODEL_DIR
    _WHISPER_MODEL = None
    _WHISPER_MODEL_SIZE = None
    _WHISPER_DEVICE = None
    _WHISPER_COMPUTE_TYPE = None
    _COSYVOICE_MODEL = None
    _COSYVOICE_MODEL_DIR = None


# ----------------------------------------------------------------------------
# 1. download_video
# ----------------------------------------------------------------------------


def download_video(
    url: str,
    output_dir: str,
    max_duration_sec: int = 60,
) -> dict:
    """Download a YouTube Shorts / Instagram Reels URL to disk.

    Uses yt-dlp to fetch the best video+audio combined stream and, via the
    `FFmpegExtractAudio` postprocessor, an MP3 sidecar. Both files end up in
    `output_dir`.

    Parameters
    ----------
    url : str
        A public Shorts or Reels URL.
    output_dir : str
        Directory to write files into. Will be created if missing.
    max_duration_sec : int
        Reject the URL if yt-dlp reports a longer duration. Default 60.

    Returns
    -------
    dict with keys:
        - "video_path": absolute path to the .mp4
        - "audio_path": absolute path to the .mp3
        - "duration": float seconds
        - "title": str
        - "uploader": str

    Raises
    ------
    ValueError
        If the URL is empty or the video is longer than `max_duration_sec`.
    RuntimeError
        If yt-dlp cannot fetch the video for any reason (network, auth, geo).
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"download_video: invalid url: {url!r}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    import yt_dlp  # type: ignore

    # 1) Probe metadata only (no download) so we can check duration cheaply.
    probe_opts = {"quiet": True, "no_warnings": True, "extract_flat": False}
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(f"download_video: yt-dlp failed to probe {url}: {e}") from e

    duration = info.get("duration") or 0
    if duration <= 0:
        raise RuntimeError(
            f"download_video: could not determine duration for {url} "
            f"(private video? region-locked?)"
        )
    if duration > max_duration_sec:
        raise ValueError(
            f"download_video: video is {duration}s, exceeds max {max_duration_sec}s"
        )

    # 2) Real download. We use a custom outtmpl that pulls `output_dir` in so
    # the final filename is `<id>.mp4` inside output_dir.
    dl_opts = {
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "format": "bv*+ba/b",  # best separate video+audio, fallback to best muxed
        "merge_output_format": "mp4",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            },
        ],
        "quiet": True,
        "no_warnings": True,
        # Hygiene: avoids hammering the host
        "sleep_interval_requests": 1,
        "max_sleep_interval_requests": 3,
    }
    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            downloaded_info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise RuntimeError(
            f"download_video: yt-dlp failed to download {url}: {e}"
        ) from e

    # Resolve the actual file paths yt-dlp wrote.
    # `requested_downloads` is the modern location; fall back to id+ext for
    # older yt-dlp builds.
    rd = downloaded_info.get("requested_downloads") or []
    video_path = None
    if rd:
        video_path = rd[0].get("filepath") or rd[0].get("_filename")
    if not video_path:
        ext = downloaded_info.get("ext", "mp4")
        video_path = os.path.join(output_dir, f"{downloaded_info['id']}.{ext}")
    video_path = os.path.abspath(video_path)
    audio_path = os.path.splitext(video_path)[0] + ".mp3"

    if not os.path.isfile(video_path):
        raise RuntimeError(
            f"download_video: expected video at {video_path} but it's missing"
        )
    if not os.path.isfile(audio_path):
        # Some formats don't produce a separate audio sidecar; fall back to
        # extracting it ourselves.
        log.warning("download_video: sidecar mp3 missing; extracting via ffmpeg")
        _ffmpeg_extract_audio(video_path, audio_path)

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "duration": float(duration),
        "title": downloaded_info.get("title", ""),
        "uploader": downloaded_info.get("uploader")
        or downloaded_info.get("channel", ""),
    }


def _ffmpeg_extract_audio(video_path: str, audio_path: str) -> None:
    """Fallback: extract mono 16 kHz mp3 from a video file with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "192k",
        audio_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ----------------------------------------------------------------------------
# 2. transcribe_audio
# ----------------------------------------------------------------------------


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
) -> list[dict]:
    """Transcribe an audio file with faster-whisper.

    Parameters
    ----------
    audio_path : str
        Path to an audio file readable by ffmpeg (mp3/wav/m4a/...).
    model_size : str
        faster-whisper model size. `small` is a good T4 default; `base` is
        faster but worse; `large-v3` is best quality but ~3 GB VRAM.
    device : str
        `cuda` for GPU, `cpu` for fallback.
    compute_type : str
        `float16` on T4; `int8` on CPU.
    language : str, optional
        Force a source language (e.g., "en") to skip auto-detection.

    Returns
    -------
    list of dicts:
        [{"start": float, "end": float, "text": str}, ...]

    Raises
    ------
    RuntimeError
        If the audio file is missing or the model cannot load.
    """
    if not os.path.isfile(audio_path):
        raise RuntimeError(f"transcribe_audio: audio file not found: {audio_path}")

    model = _get_whisper(model_size, device, compute_type)
    log.info("transcribe_audio: transcribing %s (model=%s)", audio_path, model_size)

    try:
        segs_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,  # skip silent stretches
            word_timestamps=False,  # we only need segment-level timing
            language=language,
        )
        # `seg_iter` is a generator — must be materialised.
        segs = [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segs_iter
            if s.text and s.text.strip()
        ]
    except Exception as e:
        raise RuntimeError(f"transcribe_audio: faster-whisper failed: {e}") from e

    log.info(
        "transcribe_audio: detected language=%s (p=%.2f), %d segments",
        info.language,
        info.language_probability,
        len(segs),
    )
    return segs


# ----------------------------------------------------------------------------
# 3. translate_segments
# ----------------------------------------------------------------------------

_TRANSLATION_SYSTEM = (
    "You are a professional dubbing translator. Translate the user's text into "
    "{target_lang}. Keep the translated length within roughly ±20 percent of the "
    "original character count so the dubbed speech fits the original timing. "
    "Do not add commentary, explanations, or quotes. Output ONLY the translated "
    "sentence."
)


def translate_segments(
    segments: list[dict],
    target_lang: str,
    ollama_model: str = "llama3.1:8b",
) -> list[dict]:
    """Translate each segment's text into `target_lang` using a local Ollama LLM.

    Per-segment translation (not batched) so each segment's text length budget
    is judged individually. If the LLM call fails for a segment, that segment
    falls back to its original text and a warning is logged — the pipeline
    never crashes on a single bad translation.

    Parameters
    ----------
    segments : list[dict]
        Output of `transcribe_audio`.
    target_lang : str
        BCP-47 / Ollama language name. Examples: "Spanish", "Hindi",
        "Japanese", "French", "German", "Mandarin".
    ollama_model : str
        Ollama model tag. Default `llama3.1:8b`.

    Returns
    -------
    list of dicts, same order/length as `segments`:
        [{"start", "end", "original_text", "text"}, ...]
    """
    if not segments:
        return []
    if not target_lang:
        raise ValueError("translate_segments: target_lang is required")

    client = _get_ollama()
    system_prompt = _TRANSLATION_SYSTEM.format(target_lang=target_lang)
    out: list[dict] = []
    for i, seg in enumerate(segments):
        original = seg.get("text", "").strip()
        if not original:
            out.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "original_text": "",
                    "text": "",
                }
            )
            continue
        try:
            resp = client.chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": original},
                ],
                options={"temperature": 0.2},
            )
            translated = (resp.get("message", {}).get("content") or "").strip()
            # Strip stray quotes the model sometimes wraps answers in.
            translated = translated.strip('"').strip("'").strip("「」").strip()
            if not translated:
                raise RuntimeError("empty response")
        except Exception as e:
            log.warning(
                "translate_segments: segment %d failed (%s); keeping original text",
                i,
                e,
            )
            translated = original
        out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "original_text": original,
                "text": translated,
            }
        )
    log.info(
        "translate_segments: translated %d segments into %s", len(out), target_lang
    )
    return out


# ----------------------------------------------------------------------------
# 4. extract_reference_voice
# ----------------------------------------------------------------------------


def extract_reference_voice(
    audio_path: str,
    duration_sec: int = 8,
    output_dir: Optional[str] = None,
) -> str:
    """Extract a clean N-second clip from the original audio for voice cloning.

    Default behaviour: take a clip from the *middle* of the audio (most
    intros/outros have music or silence) and convert it to 24 kHz mono WAV
    (CosyVoice3's expected reference format).

    Parameters
    ----------
    audio_path : str
        Path to the source audio (mp3/m4a/wav/...).
    duration_sec : int
        Length of the reference clip in seconds.
    output_dir : str, optional
        Where to write the clip. Defaults to a sibling `_ref` subdir of the
        audio file.

    Returns
    -------
    str
        Absolute path to the 24 kHz mono WAV reference clip.

    Raises
    ------
    RuntimeError
        If the source audio is shorter than `duration_sec`.
    """
    if not os.path.isfile(audio_path):
        raise RuntimeError(f"extract_reference_voice: audio not found: {audio_path}")
    if duration_sec <= 0:
        raise ValueError("extract_reference_voice: duration_sec must be > 0")

    src_dir = os.path.dirname(os.path.abspath(audio_path))
    out_dir = output_dir or os.path.join(src_dir, "_ref")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.abspath(os.path.join(out_dir, "reference_24k_mono.wav"))

    total = _ffprobe_duration(audio_path)
    if total < duration_sec:
        raise RuntimeError(
            f"extract_reference_voice: audio is {total:.1f}s but {duration_sec}s requested"
        )
    # Centre the window. For very long audio this picks a representative
    # stretch; for short audio the centre is the only sensible choice.
    start = max(0.0, (total - duration_sec) / 2.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        audio_path,
        "-t",
        str(duration_sec),
        "-ac",
        "1",
        "-ar",
        "24000",
        "-codec:a",
        "pcm_s16le",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"extract_reference_voice: ffmpeg failed: {e.stderr.decode(errors='ignore')}"
        ) from e

    log.info(
        "extract_reference_voice: wrote %s (%.1fs, 24k mono)", out_path, duration_sec
    )
    return out_path


def _ffprobe_duration(path: str) -> float:
    """Return audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(json.loads(out.stdout)["format"]["duration"])


# ----------------------------------------------------------------------------
# 5. synthesize_dubbed_audio
# ----------------------------------------------------------------------------


def synthesize_dubbed_audio(
    segments: list[dict],
    reference_voice_path: str,
    output_dir: str,
    model_dir: Optional[str] = None,
) -> list[dict]:
    """Synthesize one audio file per segment using CosyVoice3 cross-lingual
    voice cloning.

    Uses `inference_cross_lingual`, which is the canonical CosyVoice3 path
    for source-LANG audio → target-LANG speech (and avoids needing the
    transcribed prompt_text that `inference_zero_shot` requires).

    Parameters
    ----------
    segments : list[dict]
        Output of `translate_segments` (must have a `text` field per item).
    reference_voice_path : str
        Path to a 24 kHz mono WAV clip of the original speaker.
    output_dir : str
        Where to write per-segment WAVs.
    model_dir : str, optional
        Override the CosyVoice model dir (default: env `COSYVOICE_MODEL_DIR`).

    Returns
    -------
    list of dicts (one per input segment) with added fields:
        "audio_path"   : str  absolute path to the synthesized clip
        "synth_duration" : float  measured length of the synthesized clip

    Raises
    ------
    PipelineError
        If the CosyVoice repo isn't found on disk or the model can't load.
    RuntimeError
        If any individual segment fails to synthesize (after a single retry).
    """
    if not segments:
        return []
    if not os.path.isfile(reference_voice_path):
        raise RuntimeError(
            f"synthesize_dubbed_audio: reference voice not found: {reference_voice_path}"
        )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_dir = model_dir or COSYVOICE_MODEL_DIR
    try:
        cv = _get_cosyvoice(model_dir)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(
            f"synthesize_dubbed_audio: failed to load CosyVoice from {model_dir}: {e}"
        ) from e

    import torchaudio  # type: ignore  # used by AutoModel.inference_* callers

    out: list[dict] = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        seg_out_path = os.path.abspath(os.path.join(output_dir, f"seg_{i:04d}.wav"))
        if not text:
            # Nothing to say for this segment — write a tiny silent clip so
            # the time-align stage has a file to work with.
            _write_silence_wav(seg_out_path, max(0.1, seg["end"] - seg["start"]))
            synth_dur = max(0.1, seg["end"] - seg["start"])
        else:
            try:
                synth_dur = _cosyvoice_synth_one(
                    cv, text, reference_voice_path, seg_out_path
                )
            except Exception as e:
                log.warning(
                    "synthesize_dubbed_audio: seg %d failed (%s); retrying", i, e
                )
                try:
                    synth_dur = _cosyvoice_synth_one(
                        cv, text, reference_voice_path, seg_out_path
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"synthesize_dubbed_audio: seg {i} failed twice: {e2}"
                    ) from e2

        out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "original_text": seg.get("original_text", ""),
                "text": text,
                "audio_path": seg_out_path,
                "synth_duration": float(synth_dur),
            }
        )
    log.info("synthesize_dubbed_audio: produced %d clips", len(out))
    return out


def _cosyvoice_synth_one(cv, text: str, ref_audio: str, out_path: str) -> float:
    """Run a single CosyVoice3 cross-lingual inference and save the result.
    Returns the duration of the saved clip in seconds."""
    import torchaudio  # type: ignore

    # inference_cross_lingual returns a generator of dicts with `tts_speech`.
    # In non-streaming mode there's exactly one chunk for short inputs.
    for chunk in cv.inference_cross_lingual(
        tts_text=text, prompt_audio=ref_audio, stream=False
    ):
        wav = chunk["tts_speech"]  # torch.Tensor, shape (1, samples)
        torchaudio.save(out_path, wav.cpu(), cv.sample_rate)
        samples = wav.shape[-1]
        return float(samples) / float(cv.sample_rate)
    raise RuntimeError("inference_cross_lingual yielded no chunks")


def _write_silence_wav(path: str, duration_sec: float) -> None:
    """Write a mono 24 kHz silent WAV of the given length."""
    sr = 24000
    n = max(1, int(duration_sec * sr))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n)


# ----------------------------------------------------------------------------
# 6. time_align_segment
# ----------------------------------------------------------------------------


def _split_atempo(ratio: float) -> list[float]:
    """Factor a stretch ratio into a list of atempo values, each in [0.5, 2.0].

    ffmpeg's `atempo` filter only accepts values between 0.5 and 2.0. To
    compress by 4x (ratio 0.25) we chain four 0.5 atempo filters, etc.

    Strategy
    --------
    - ratio > 1 (speed up): fill with 2.0's then a residual.
    - ratio < 1 (slow down): fill with 0.5's then a residual > 0.5.
    - Output order: heaviest change first, gentlest last (recommended by
      ffmpeg docs for best quality — last filter sees the most natural signal).
    """
    if ratio <= 0:
        raise ValueError(f"_split_atempo: ratio must be > 0, got {ratio}")
    if ratio == 1.0:
        return [1.0]

    factors: list[float] = []
    if ratio > 1.0:
        n = math.ceil(math.log2(ratio))
        for _ in range(n):
            factors.append(2.0)
            ratio /= 2.0
        factors.append(ratio)  # residual < 2.0
        factors.reverse()  # heaviest first
    else:
        inv = 1.0 / ratio
        n = math.ceil(math.log2(inv))
        for _ in range(n):
            factors.append(0.5)
            ratio *= 2.0  # track "what's still needed to be 1.0"
        factors.append(ratio)  # residual in (0.5, 1.0]
    # Clamp to legal range just in case of float drift.
    return [max(0.5, min(2.0, f)) for f in factors]


def time_align_segment(
    audio_path: str,
    target_duration: float,
    output_path: str,
) -> str:
    """Time-stretch `audio_path` so the result is exactly `target_duration`
    seconds long, using ffmpeg's `atempo` filter (chained if necessary).

    Parameters
    ----------
    audio_path : str
        Input WAV (any sample rate / channels; ffmpeg normalises).
    target_duration : float
        Desired output length in seconds.
    output_path : str
        Where to write the aligned WAV.

    Returns
    -------
    str
        The `output_path` argument, for chaining.

    Raises
    ------
    ValueError
        If `target_duration` is non-positive.
    subprocess.CalledProcessError
        If ffmpeg fails (e.g., missing input file).
    """
    if not os.path.isfile(audio_path):
        raise RuntimeError(f"time_align_segment: input not found: {audio_path}")
    if target_duration <= 0:
        raise ValueError(
            f"time_align_segment: target_duration must be > 0, got {target_duration}"
        )

    actual = _ffprobe_duration(audio_path)
    # Guard against divide-by-zero on degenerate clips.
    if actual <= 0.001:
        # Just write silence of the right length.
        _write_silence_wav(output_path, target_duration)
        log.warning("time_align_segment: input %s is empty; wrote silence", audio_path)
        return output_path

    ratio = target_duration / actual  # > 1 = slow down (synth too short)
    factors = _split_atempo(ratio)
    filter_str = ",".join(f"atempo={f:.6f}" for f in factors)

    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-filter:a",
        filter_str,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "24000",
        "-codec:a",
        "pcm_s16le",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"time_align_segment: ffmpeg failed for {audio_path}: "
            f"{e.stderr.decode(errors='ignore')}"
        ) from e

    log.info(
        "time_align_segment: %s -> %s (%.3fs -> %.3fs, ratio=%.3f, factors=%s)",
        os.path.basename(audio_path),
        os.path.basename(output_path),
        actual,
        target_duration,
        ratio,
        factors,
    )
    return output_path


# ----------------------------------------------------------------------------
# 7. mux_final_video
# ----------------------------------------------------------------------------


def mux_final_video(
    video_path: str,
    aligned_segments: list[dict],
    output_path: str,
) -> str:
    """Replace the original video's audio track with the dubbed audio.

    Each aligned segment is placed at its original `start` time; gaps are
    preserved as silence. The video stream is copied (no re-encode) for
    speed and quality.

    Parameters
    ----------
    video_path : str
        Path to the source .mp4 (untouched audio replaced).
    aligned_segments : list[dict]
        Each item must have: "audio_path" (str), "start" (float seconds).
        Typically the output of `time_align_segment` calls.
    output_path : str
        Where to write the final .mp4.

    Returns
    -------
    str
        `output_path`.

    Raises
    ------
    RuntimeError / subprocess.CalledProcessError
        On ffmpeg failure.
    """
    if not os.path.isfile(video_path):
        raise RuntimeError(f"mux_final_video: video not found: {video_path}")
    if not aligned_segments:
        raise ValueError("mux_final_video: aligned_segments is empty")

    # Build the filter_complex graph:
    #   [0:a]volume=0            -> original audio, muted (we'll concat over it)
    #   [1:a]adelay=DELAY|aDELAY -> per-segment delayed dubbed audio
    #   [m0][m1]...amix          -> mix with the muted original? No, we want to
    #                              REPLACE. Use `concat` instead.
    #
    # Graph:
    #   for i in segments:  [i+1:a]adelay=<ms>|<ms>[d{i}]
    #   [d0][d1]...[dN-1]concat=n=N:v=0:a=1[out]
    n = len(aligned_segments)
    inputs = ["-i", video_path]
    label_idx = 1
    delay_labels: list[str] = []
    for seg in aligned_segments:
        ap = seg.get("audio_path")
        if not ap or not os.path.isfile(ap):
            raise RuntimeError(f"mux_final_video: segment audio missing: {ap}")
        inputs.extend(["-i", ap])
        delay_ms = int(round(float(seg["start"]) * 1000))
        delay_labels.append(f"d{label_idx - 1}")
        label_idx += 1

    # Per-input adelay chains.
    filter_parts: list[str] = []
    for idx, seg in enumerate(aligned_segments):
        delay_ms = int(round(float(seg["start"]) * 1000))
        # adelay takes per-channel delays separated by '|'; we use 1 channel.
        filter_parts.append(f"[{idx + 1}:a]adelay={delay_ms}|{delay_ms},apad[d{idx}]")
    # Concat all delayed streams.
    concat_inputs = "".join(f"[d{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=0:a=1[outa]")
    filter_complex = ";\n".join(filter_parts)

    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v",  # original video stream
        "-map",
        "[outa]",  # new dubbed audio
        "-c:v",
        "copy",  # no re-encode
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",  # cut at the shorter of the two streams
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"mux_final_video: ffmpeg failed: {e.stderr.decode(errors='ignore')}"
        ) from e

    log.info("mux_final_video: wrote %s", output_path)
    return output_path


# ----------------------------------------------------------------------------
# 8. run_pipeline
# ----------------------------------------------------------------------------


@dataclass
class PipelineResult:
    final_video_path: str
    output_dir: str
    target_lang: str
    n_segments: int


def run_pipeline(
    url: str,
    target_lang: str,
    output_dir: str,
    ollama_model: str = "llama3.1:8b",
    whisper_model_size: str = "small",
    whisper_device: str = "cuda",
    whisper_compute_type: str = "float16",
    reference_duration_sec: int = 8,
    max_duration_sec: int = 60,
) -> str:
    """End-to-end dubbing pipeline. Runs all 7 stages in order with
    per-stage try/except and clear error messages.

    Returns the final dubbed video path. Raises PipelineError on any failure.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    work = os.path.join(output_dir, "work")
    Path(work).mkdir(parents=True, exist_ok=True)

    def _stage(name: str, fn: Callable[[], Any]) -> Any:
        log.info("=== STAGE: %s ===", name)
        try:
            return fn()
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"[stage: {name}] failed: {e}") from e

    # Stage 1: download
    dl = _stage(
        "1/7 download_video",
        lambda: download_video(
            url=url,
            output_dir=work,
            max_duration_sec=max_duration_sec,
        ),
    )
    video_path = dl["video_path"]
    audio_path = dl["audio_path"]
    log.info("downloaded: %s (%.1fs)", dl["title"], dl["duration"])

    # Stage 2: transcribe
    raw_segs = _stage(
        "2/7 transcribe_audio",
        lambda: transcribe_audio(
            audio_path=audio_path,
            model_size=whisper_model_size,
            device=whisper_device,
            compute_type=whisper_compute_type,
        ),
    )
    if not raw_segs:
        raise PipelineError("[stage: 2/7 transcribe_audio] returned 0 segments")

    # Stage 3: translate
    translated = _stage(
        "3/7 translate_segments",
        lambda: translate_segments(
            segments=raw_segs,
            target_lang=target_lang,
            ollama_model=ollama_model,
        ),
    )

    # Stage 4: reference voice clip
    ref_path = _stage(
        "4/7 extract_reference_voice",
        lambda: extract_reference_voice(
            audio_path=audio_path,
            duration_sec=reference_duration_sec,
            output_dir=work,
        ),
    )

    # Stage 5: synthesise (slow — this is where CosyVoice3 lives)
    synth = _stage(
        "5/7 synthesize_dubbed_audio",
        lambda: synthesize_dubbed_audio(
            segments=translated,
            reference_voice_path=ref_path,
            output_dir=os.path.join(work, "synth"),
        ),
    )

    # Stage 6: time-align each segment to its original duration
    aligned_dir = os.path.join(work, "aligned")
    Path(aligned_dir).mkdir(parents=True, exist_ok=True)
    aligned: list[dict] = []

    def _align_all() -> list[dict]:
        out: list[dict] = []
        for i, s in enumerate(synth):
            target = max(0.05, float(s["end"]) - float(s["start"]))
            aligned_path = os.path.abspath(
                os.path.join(aligned_dir, f"aligned_{i:04d}.wav")
            )
            time_align_segment(
                audio_path=s["audio_path"],
                target_duration=target,
                output_path=aligned_path,
            )
            out.append(
                {
                    "start": s["start"],
                    "end": s["end"],
                    "audio_path": aligned_path,
                }
            )
        return out

    aligned = _stage("6/7 time_align_segment", _align_all)

    # Stage 7: mux
    final_path = os.path.abspath(os.path.join(output_dir, f"dubbed_{target_lang}.mp4"))
    _stage(
        "7/7 mux_final_video",
        lambda: mux_final_video(
            video_path=video_path,
            aligned_segments=aligned,
            output_path=final_path,
        ),
    )

    log.info("run_pipeline: SUCCESS -> %s", final_path)
    return final_path


# ----------------------------------------------------------------------------
# CLI smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    p = argparse.ArgumentParser(description="YouDub CLI smoke test")
    p.add_argument("url", help="YouTube Shorts or Instagram Reel URL")
    p.add_argument("target_lang", help="Target language, e.g. Spanish, Hindi")
    p.add_argument("--out", default="./youdub_out", help="Output directory")
    p.add_argument("--ollama-model", default="llama3.1:8b")
    p.add_argument("--whisper-size", default="small")
    args = p.parse_args()

    out = run_pipeline(
        url=args.url,
        target_lang=args.target_lang,
        output_dir=args.out,
        ollama_model=args.ollama_model,
        whisper_model_size=args.whisper_size,
    )
    print(f"\nFinal dubbed video: {out}")
