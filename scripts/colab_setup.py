"""
YouDub Colab setup script.

This is the SINGLE source of truth for Colab environment setup. It replaces
the previous 11 inline notebook cells so that fixes only need to happen
in this repo (not in the Colab notebook). Re-running the script is safe;
each stage has its own idempotency check.

Two-environment architecture
----------------------------
Colab's main kernel is Python 3.13, which has no wheel for CosyVoice3's
exact pins (torch==2.3.1, numpy==1.26.4, onnxruntime-gpu==1.18.0,
openai-whisper==20231117). Rather than relax pins (which risks Qwen2
breakage) or pin Colab to Py3.12 (not viable via the Colab UI), this
script isolates CosyVoice in its own Python 3.11 venv on Drive:

  - The main Colab kernel (3.13) runs the pipeline orchestration,
    faster-whisper, Ollama, ffmpeg, and all 8 pipeline functions except
    the TTS model call.
  - A Python 3.11 venv at $DRIVE_ROOT/cosyv_venv311 holds CosyVoice's
    original unmodified requirements.txt.
  - pipeline.py's synthesize_dubbed_audio spawns a long-running
    scripts/cosyv_infer.py subprocess in the venv and talks to it
    over a newline-delimited JSON protocol on stdin/stdout.

Usage from a Colab cell:

    !python "{DRIVE_ROOT}/scripts/colab_setup.py"
    !source  "{DRIVE_ROOT}/scripts/colab_env.sh"   # export env vars to this cell

CLI flags (all default to "do the thing"; flags are for power users):

    --drive-root PATH    Override the YouDub project root (default: $DRIVE_ROOT
                         from env, or /content/drive/MyDrive/YouDub).
    --ollama-host URL    Set OLLAMA_HOST (default: http://localhost:11434).
    --hf-model-id ID     HF repo to download CosyVoice3 weights from.
    --skip-venv          Skip the Python 3.11 venv creation/refresh.
    --skip-system-deps   Skip apt-get install.
    --skip-pip           Skip pip install of light deps.
    --skip-clone         Skip CosyVoice git clone/pull.
    --skip-weights       Skip HF weights download.
    --skip-cosyv-deps    Skip pip install of CosyVoice's pinned requirements.
    --cosyvoice-branch   Git branch/tag to checkout (default: main).

Stages
------
1. Resolve & export paths (DRIVE_ROOT, COSYVOICE_REPO_DIR, COSYVOICE_MODEL_DIR,
   COSYV_VENV_PYTHON, OLLAMA_HOST).
2. apt-install Python 3.11 + create the venv at $DRIVE_ROOT/cosyv_venv311.
3. System deps (ffmpeg, sox, git-lfs).
4. Light Python deps (yt-dlp, faster-whisper, ollama, etc.) into the main env.
5. Clone or update FunAudioLLM/CosyVoice (with submodules).
6. Snapshot-download CosyVoice3 weights from Hugging Face.
7. Install CosyVoice's unmodified requirements.txt into the venv.
8. Sanity-check: import AutoModel from the cloned CosyVoice repo via the venv.

After all stages, writes scripts/colab_env.sh next to this file with the
exported env vars so a Colab cell can `source` it and inherit them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults — keep in sync with pipeline.py's module-level env-var fallbacks.
# ---------------------------------------------------------------------------

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/YouDub"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_HF_MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
DEFAULT_COSYVOICE_REPO_URL = "https://github.com/FunAudioLLM/CosyVoice.git"
DEFAULT_COSYVOICE_MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B"  # no "-2512" suffix on disk
DEFAULT_COSYVOICE_BRANCH = "main"

# Python 3.11 is the last version with full wheel support for CosyVoice3's
# pinned torch 2.3.1, numpy 1.26.4, onnxruntime-gpu 1.18.0,
# openai-whisper 20231117. Colab's default kernel is 3.13 and we don't
# try to change that — we install a 3.11 venv on Drive and route CosyVoice
# work to it via a subprocess.
PY311_VERSION = "3.11"
COSYV_VENV_DIRNAME = "cosyv_venv311"
COSYV_VENV_MARKER = ".youdub_venv_ready"
COSYV_VENV_PATH_JSON = "venv_python_path.json"
COSYV_INFER_SCRIPT_NAME = "cosyv_infer.py"

LIGHT_PIP_DEPS = [
    "yt-dlp>=2024.8.6",
    "faster-whisper>=1.0.3",
    "ffmpeg-python>=0.2.0",
    "huggingface_hub>=0.24.0",
    "ollama>=0.3.0",
    "requests>=2.32.0",
    "ipywidgets",
]

APT_PACKAGES = ["ffmpeg", "sox", "libsox-dev", "git-lfs"]

# When a subprocess fails, _run() prints the last N characters of its merged
# stdout/stderr so the real error is visible without a manual re-run. 3000
# is enough to cover pip's full error message + a few dozen progress-bar
# lines of context, without flooding the log on a 9.75 GB download.
_FAILURE_OUTPUT_TAIL_CHARS = 3000


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _log(stage: str, msg: str) -> None:
    """One-line stage-prefixed log."""
    print(f"[{stage}] {msg}", flush=True)


def _print_failure_output(pretty_cmd: str, output: str) -> None:
    """Print the tail of a failed subprocess's output so the user can see
    the real error without re-running the command. Defensive against
    UnicodeEncodeError so a single weird byte can't crash the script.
    """
    n = len(output)
    if n == 0:
        print(f"  >>> command failed with no captured output: {pretty_cmd}", flush=True)
        return
    if n > _FAILURE_OUTPUT_TAIL_CHARS:
        skipped = n - _FAILURE_OUTPUT_TAIL_CHARS
        header = (
            f"  >>> command failed: {pretty_cmd}\n"
            f"  >>> {n} bytes of output; showing last "
            f"{_FAILURE_OUTPUT_TAIL_CHARS} (skipped {skipped})"
        )
        body = "...<truncated>...\n" + output[-_FAILURE_OUTPUT_TAIL_CHARS:]
    else:
        header = f"  >>> command failed: {pretty_cmd}\n  >>> {n} bytes of output:"
        body = output
    print(header, flush=True)
    print("  ─── begin output ───", flush=True)
    try:
        print(body, end="" if body.endswith("\n") else "\n", flush=True)
    except UnicodeEncodeError:
        # Some Colab sessions have a non-UTF stdout codec. Encode manually
        # with 'replace' so we never crash the script just to print an
        # error message about a different error.
        sys.stdout.buffer.write(
            (body + ("" if body.endswith("\n") else "\n")).encode(
                sys.stdout.encoding or "utf-8", errors="replace"
            )
        )
        sys.stdout.flush()
    print("  ─── end output ───", flush=True)


def _run(
    cmd: list[str], *, cwd: Optional[str] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """subprocess.run with a printed command and self-diagnosing failures.

    On non-zero exit with check=True, prints the last
    _FAILURE_OUTPUT_TAIL_CHARS of the captured output (stdout merged with
    stderr) before re-raising CalledProcessError, so the real error is
    visible directly in the script's log.
    """
    pretty = " ".join(cmd) if isinstance(cmd, list) else cmd
    if cwd:
        pretty = f"(cwd={cwd}) {pretty}"
    print(f"  $ {pretty}", flush=True)
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        # e.output is the merged stdout+stderr (because we used STDOUT above)
        # when text=True is set; on older Python or non-text mode, fall back
        # to e.output.decode best-effort.
        output = e.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        _print_failure_output(pretty, output)
        raise


def _which(binary: str) -> Optional[str]:
    """Return path to binary or None."""
    from shutil import which

    return which(binary)


def _is_root() -> bool:
    """True if running as root. Handles non-POSIX platforms (Windows).

    Uses getattr with a default rather than hasattr+attribute because
    some static analyzers don't narrow the type from hasattr().
    """
    geteuid = getattr(os, "geteuid", None)
    return geteuid is not None and geteuid() == 0


def _pip_installed(pkg: str) -> bool:
    """True if `pip show pkg` succeeds (i.e. the package is importable enough)."""
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "show", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _filter_requirements_txt(path: str, drop_pkgs: set[str]) -> str:
    """Return the path to a temp file containing `path` with lines whose
    first non-whitespace token's package name matches one of `drop_pkgs`
    removed. Preserves comments, blank lines, --extra-index-url, and
    unrelated packages. Caller is responsible for cleaning up the file.

    Used to skip packages that need special install handling (e.g.,
    `--no-build-isolation`) from a bulk `pip install -r` call. The
    package-name match is case-insensitive and stops at the first
    version/specifier character (==, >=, ~=, ;, ,).
    """
    import re
    import tempfile

    drop = {p.lower() for p in drop_pkgs}
    pattern = re.compile(r"^\s*([A-Za-z0-9_.+-]+)")
    keep: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m and m.group(1).lower() in drop:
                continue
            keep.append(line)
    fd, out = tempfile.mkstemp(
        prefix="youdub_cosyv_filtered_", suffix=".txt", text=True
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(keep)
    return out


# ---------------------------------------------------------------------------
# Stage 1: paths & env vars
# ---------------------------------------------------------------------------


def stage_paths(args: argparse.Namespace) -> dict[str, str]:
    """Resolve paths and write them to a colab_env.sh next to this script.

    The .sh file lets a subsequent Colab cell `source` the env vars even
    though the script ran in a subshell.
    """
    drive_root = args.drive_root or os.environ.get("DRIVE_ROOT") or DEFAULT_DRIVE_ROOT
    drive_root = os.path.abspath(drive_root)
    Path(drive_root).mkdir(parents=True, exist_ok=True)

    cosy_repo = os.path.join(drive_root, "CosyVoice")
    cosy_model = os.path.join(
        cosy_repo, "pretrained_models", DEFAULT_COSYVOICE_MODEL_DIR_NAME
    )
    cosy_venv_dir = os.path.join(drive_root, COSYV_VENV_DIRNAME)
    # The venv's python path is only known after stage_create_cosy_venv
    # runs, so we leave it as a placeholder here and let that stage update
    # the env file with the real path.
    cosy_venv_python = os.path.join(cosy_venv_dir, "bin", "python")
    ollama_host = (
        args.ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    )

    env_vars = {
        "DRIVE_ROOT": drive_root,
        "COSYVOICE_REPO_DIR": cosy_repo,
        "COSYVOICE_MODEL_DIR": cosy_model,
        "COSYV_VENV_DIR": cosy_venv_dir,
        "COSYV_VENV_PYTHON": cosy_venv_python,
        "OLLAMA_HOST": ollama_host,
        "YOUDUB_HF_MODEL_ID": args.hf_model_id,
    }
    for k, v in env_vars.items():
        os.environ[k] = v

    # Write colab_env.sh next to this script so a Colab cell can `source` it.
    env_sh_path = Path(__file__).resolve().parent / "colab_env.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by scripts/colab_setup.py — do not edit by hand.",
        "# Re-run colab_setup.py to refresh.",
    ]
    for k, v in env_vars.items():
        lines.append(f'export {k}="{v}"')
    env_sh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log("1/8 paths", f"DRIVE_ROOT={drive_root}")
    _log("1/8 paths", f"COSYVOICE_REPO_DIR={cosy_repo}")
    _log("1/8 paths", f"COSYVOICE_MODEL_DIR={cosy_model}")
    _log(
        "1/8 paths", f"COSYV_VENV_PYTHON={cosy_venv_python} (will exist after Stage 2)"
    )
    _log("1/8 paths", f"OLLAMA_HOST={ollama_host}")
    _log("1/8 paths", f"wrote {env_sh_path}")
    return env_vars


# ---------------------------------------------------------------------------
# Stage 2: apt-install Python 3.11 + create the venv on Drive
# ---------------------------------------------------------------------------


def _apt_install_python311() -> Optional[str]:
    """apt-install python3.11 + python3.11-venv + python3.11-dev. Returns
    the path to the python3.11 binary (e.g. /usr/bin/python3.11) on
    success, or None on failure.

    Tries three strategies in order:
      1. apt-get install python3.11  (works on Colab Ubuntu 22.04 as of 2025)
      2. install software-properties-common, add deadsnakes PPA, retry
      3. give up and return None
    """
    use_sudo = _which("sudo") is not None and not _is_root()
    apt = ["sudo"] if use_sudo else []

    candidates = [
        f"python{PY311_VERSION}",
        f"python{PY311_VERSION}-venv",
        f"python{PY311_VERSION}-dev",
    ]

    def _have(exe: str) -> bool:
        return _which(exe) is not None

    # Strategy 1: direct apt install
    if not _have(candidates[0]):
        _log("2/8 venv", f"trying apt install {candidates}")
        try:
            _run([*apt, "apt-get", "update", "-qq"])
            _run([*apt, "apt-get", "install", "-y", "-qq", *candidates])
        except Exception as e:
            _log("2/8 venv", f"direct apt install failed: {e!r}")
        else:
            if _have(candidates[0]):
                _log("2/8 venv", f"apt install OK; {candidates[0]} now on PATH")
                return _which(candidates[0])

    # Strategy 2: deadsnakes PPA
    if not _have(candidates[0]):
        _log("2/8 venv", "Python 3.11 not on PATH; trying deadsnakes PPA")
        try:
            _run(
                [*apt, "apt-get", "install", "-y", "-qq", "software-properties-common"]
            )
            _run([*apt, "add-apt-repository", "-y", "ppa:deadsnakes/ppa"])
            _run([*apt, "apt-get", "update", "-qq"])
            _run([*apt, "apt-get", "install", "-y", "-qq", *candidates])
        except Exception as e:
            _log("2/8 venv", f"deadsnakes PPA path failed: {e!r}")
            return None
        if _have(candidates[0]):
            _log("2/8 venv", f"deadsnakes OK; {candidates[0]} now on PATH")
            return _which(candidates[0])
    return None if not _have(candidates[0]) else _which(candidates[0])


def _venv_python_path(venv_dir: str) -> str:
    """Return the absolute path to the venv's python interpreter."""
    if sys.platform == "win32":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def stage_create_cosy_venv(args: argparse.Namespace) -> str:
    """Stage 2: install Python 3.11 and create a venv on Drive.

    Returns the absolute path to the venv's python interpreter. On a
    warm re-run, this is a no-op (marker present, venv intact).
    """
    if args.skip_venv:
        _log("2/8 venv", "skipped (--skip-venv)")
        return os.environ["COSYV_VENV_PYTHON"]

    venv_dir = os.environ["COSYV_VENV_DIR"]
    marker = os.path.join(venv_dir, COSYV_VENV_MARKER)
    py = _venv_python_path(venv_dir)
    if os.path.isfile(marker) and os.path.isfile(py):
        _log("2/8 venv", f"marker present + python exists at {py}; skipping")
        os.environ["COSYV_VENV_PYTHON"] = py
        return py

    Path(venv_dir).mkdir(parents=True, exist_ok=True)

    py311 = _apt_install_python311()
    if not py311:
        raise RuntimeError(
            f"Could not install Python {PY311_VERSION} via apt. See the log "
            f"above for which strategy failed. On Colab, this usually means "
            f"the base image is older than expected or the deadsnakes PPA "
            f"could not be reached. Try `Runtime -> Factory reset runtime` "
            f"and re-run, or check your network."
        )

    if not os.path.isfile(py):
        _log("2/8 venv", f"creating venv at {venv_dir} (one-time, ~30s)")
        _run([py311, "-m", "venv", venv_dir])

    if not os.path.isfile(py):
        raise RuntimeError(f"venv creation reported success but {py} is missing")

    # Sanity: venv's python must report the right version.
    out = subprocess.run([py, "--version"], capture_output=True, text=True)
    if PY311_VERSION not in out.stdout + out.stderr:
        raise RuntimeError(
            f"venv python at {py} reports unexpected version: "
            f"{out.stdout.strip()!r} / {out.stderr.strip()!r}"
        )
    _log("2/8 venv", f"venv python OK: {out.stdout.strip()}")

    Path(marker).write_text(
        f"Created by colab_setup.py on {os.environ.get('HOSTNAME', 'unknown')}\n"
        f"python={py}\n",
        encoding="utf-8",
    )
    os.environ["COSYV_VENV_PYTHON"] = py
    _log("2/8 venv", f"wrote marker {marker}")
    return py


# ---------------------------------------------------------------------------
# Stage 3: system deps
# ---------------------------------------------------------------------------


def stage_system_deps(args: argparse.Namespace) -> None:
    if args.skip_system_deps:
        _log("3/8 system-deps", "skipped (--skip-system-deps)")
        return
    if all(_which(p) for p in ["ffmpeg", "sox", "git-lfs"]):
        _log("3/8 system-deps", "ffmpeg, sox, git-lfs already present; skipping")
        return
    _log("3/8 system-deps", "apt-get update + install ffmpeg, sox, libsox-dev, git-lfs")
    subprocess.run(
        ["sudo", "-n", "true"], check=False, capture_output=True
    )  # noop, may fail
    # Use sudo if available and we're not already root. Colab runs as root,
    # so sudo isn't needed there. _is_root() handles non-POSIX platforms.
    use_sudo = _which("sudo") is not None and not _is_root()
    apt = ["sudo"] if use_sudo else []
    _run([*apt, "apt-get", "update", "-qq"])
    _run([*apt, "apt-get", "install", "-y", "-qq", *APT_PACKAGES])
    _run(["git", "lfs", "install"])
    # Print the ffmpeg version for the log.
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    _log("3/8 system-deps", out.stdout.splitlines()[0])


# ---------------------------------------------------------------------------
# Stage 4: light Python deps (main env, NOT the venv)
# ---------------------------------------------------------------------------


def stage_pip_light(args: argparse.Namespace) -> None:
    if args.skip_pip:
        _log("4/8 pip-light", "skipped (--skip-pip)")
        return
    missing = [
        p for p in LIGHT_PIP_DEPS if not _pip_installed(p.split(">=")[0].split("==")[0])
    ]
    if not missing:
        _log("4/8 pip-light", "all light deps already installed; skipping")
        return
    _log("4/8 pip-light", f"installing {len(missing)} missing deps")
    _run([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])


# ---------------------------------------------------------------------------
# Stage 5: clone or update CosyVoice
# ---------------------------------------------------------------------------


def stage_clone_cosyvoice(args: argparse.Namespace) -> str:
    if args.skip_clone:
        _log("5/8 clone", "skipped (--skip-clone)")
        return os.environ["COSYVOICE_REPO_DIR"]
    repo = os.environ["COSYVOICE_REPO_DIR"]
    if not os.path.isdir(repo):
        _log(
            "5/8 clone",
            f"cloning {DEFAULT_COSYVOICE_REPO_URL} -> {repo} (one-time, ~1 min)",
        )
        os.makedirs(os.path.dirname(repo), exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--recursive",
                "--branch",
                args.cosyvoice_branch,
                DEFAULT_COSYVOICE_REPO_URL,
                repo,
            ]
        )
    else:
        _log("5/8 clone", f"{repo} already exists; pulling + updating submodules")
        # Don't blow up on a failed pull (e.g. transient network) — the repo
        # is still usable if the previous clone succeeded.
        try:
            _run(["git", "-C", repo, "pull", "--recurse-submodules"], check=False)
        except Exception as e:
            _log("5/8 clone", f"pull failed (non-fatal): {e}")
    # Always run submodule update to recover from a partial previous clone.
    _run(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
    matcha = os.path.join(repo, "third_party", "Matcha-TTS")
    if not os.path.isdir(matcha):
        raise RuntimeError(
            f"Matcha-TTS submodule missing at {matcha}. Check the clone step above."
        )
    _log("5/8 clone", "Matcha-TTS present")
    return repo


# ---------------------------------------------------------------------------
# Stage 6: download CosyVoice3 weights
# ---------------------------------------------------------------------------


def stage_download_weights(args: argparse.Namespace) -> str:
    if args.skip_weights:
        _log("6/8 weights", "skipped (--skip-weights)")
        return os.environ["COSYVOICE_MODEL_DIR"]
    model_dir = os.environ["COSYVOICE_MODEL_DIR"]
    marker = os.path.join(model_dir, "cosyvoice3.yaml")
    if os.path.isdir(model_dir) and os.path.isfile(marker):
        _log("6/8 weights", f"cosyvoice3.yaml already present at {model_dir}; skipping")
        return model_dir
    _log("6/8 weights", f"snapshot_download {args.hf_model_id} -> {model_dir}")
    from huggingface_hub import snapshot_download

    snapshot_download(
        args.hf_model_id,
        local_dir=model_dir,
        max_workers=4,
    )
    if not os.path.isfile(marker):
        raise RuntimeError(
            f"Downloaded weights but {marker} is still missing. Check the HF repo id."
        )
    _log("6/8 weights", "download complete")
    # Report size.
    out = subprocess.run(["du", "-sh", model_dir], capture_output=True, text=True)
    if out.stdout.strip():
        _log("6/8 weights", f"on-disk size: {out.stdout.split()[0]}")
    return model_dir


# ---------------------------------------------------------------------------
# Stage 7: install CosyVoice's pinned requirements into the venv
# ---------------------------------------------------------------------------


def stage_install_cosyv_deps(args: argparse.Namespace, repo: str) -> None:
    """Install CosyVoice's ORIGINAL, UNMODIFIED requirements.txt into the
    Py3.11 venv. No filtering or relaxing is needed because the venv
    has full wheel support for CosyVoice's pinned versions.
    """
    if args.skip_cosyv_deps:
        _log("7/8 cosyv-deps", "skipped (--skip-cosyv-deps)")
        return
    py = os.environ["COSYV_VENV_PYTHON"]
    if not os.path.isfile(py):
        raise RuntimeError(
            f"venv python not found at {py}. Did Stage 2 (venv creation) succeed?"
        )

    # Marker lives in the venv (not in the CosyVoice repo) so it's wiped
    # only if the user nukes the venv on purpose.
    marker = os.path.join(os.environ["COSYV_VENV_DIR"], COSYV_VENV_MARKER + ".deps")
    if os.path.isfile(marker):
        _log("7/8 cosyv-deps", f"marker {marker} present; skipping")
        return

    req = os.path.join(repo, "requirements.txt")
    if not os.path.isfile(req):
        raise RuntimeError(f"CosyVoice requirements.txt missing at {req}")

    # openai-whisper==20231117's old setup.py imports `pkg_resources` at
    # build time. Two preconditions must be in place in the venv before
    # we install it:
    #   1. setuptools<80 — setuptools 80 split pkg_resources into a
    #      separate `setuptools-pkg-resources` package, so venvs created
    #      with newer setuptools don't have it and the build fails.
    #   2. wheel — required by --no-build-isolation (see below).
    # Pin both into the venv first. Cheap on warm re-runs (~2s).
    _log(
        "7/8 cosyv-deps",
        "bootstrapping setuptools<80 + wheel (build deps for --no-build-isolation)",
    )
    _run([py, "-m", "pip", "install", "-q", "setuptools<80", "wheel"])

    # Main requirements.txt install, but with openai-whisper filtered
    # out. openai-whisper is installed separately below with
    # --no-build-isolation so it builds against the venv's setuptools<80
    # rather than a fresh isolated build env (pip's default build
    # isolation would fetch the latest setuptools into a throwaway
    # env, making our setuptools<80 pin invisible to the build step).
    filtered_req = _filter_requirements_txt(req, drop_pkgs={"openai-whisper"})
    try:
        _log(
            "7/8 cosyv-deps",
            f"pip install -r {req} (openai-whisper filtered out) via venv python, 5-10 min",
        )
        # No -U: CosyVoice's pins are intentional; we want exact versions,
        # not "newest compatible". An upgrade could silently bump past a
        # version that the Qwen2 LLM path was built against.
        _run([py, "-m", "pip", "install", "-q", "-r", filtered_req], cwd=repo)
    finally:
        # Leave the temp file in place for post-mortem on failure; clean
        # up on success.
        try:
            os.unlink(filtered_req)
        except OSError:
            pass

    # openai-whisper: install separately with --no-build-isolation so
    # the build uses the venv's setuptools<80 (which has pkg_resources
    # bundled) rather than a fresh isolated env with the latest
    # setuptools (which doesn't). wheel is already in the venv from
    # the bootstrap step above.
    _log(
        "7/8 cosyv-deps",
        "pip install openai-whisper==20231117 (--no-build-isolation, uses venv setuptools<80)",
    )
    _run(
        [
            py,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-build-isolation",
            "openai-whisper==20231117",
        ]
    )

    Path(marker).write_text(
        f"Installed by colab_setup.py on {os.environ.get('HOSTNAME', 'unknown')}\n"
        f"requirements={req}\n",
        encoding="utf-8",
    )
    _log("7/8 cosyv-deps", f"wrote marker {marker}")


# ---------------------------------------------------------------------------
# Stage 8: AutoModel import sanity check (via the venv's python)
# ---------------------------------------------------------------------------


def stage_sanity_check(args: argparse.Namespace) -> None:
    """Run `from cosyvoice.cli.cosyvoice import AutoModel` in the venv's
    python. On success, write venv_python_path.json so pipeline.py can
    discover the interpreter without re-running the setup script.
    """
    if args.skip_cosyv_deps:
        _log("8/8 sanity", "skipped (--skip-cosyv-deps)")
        return
    py = os.environ["COSYV_VENV_PYTHON"]
    if not os.path.isfile(py):
        raise RuntimeError(
            f"venv python not found at {py}. Did Stage 2 (venv creation) succeed?"
        )
    repo = os.environ["COSYVOICE_REPO_DIR"]
    matcha = os.path.join(repo, "third_party", "Matcha-TTS")
    _log("8/8 sanity", f"importing AutoModel via venv python {py}")
    code = (
        "import sys; "
        f"sys.path.insert(0, {repo!r}); "
        f"sys.path.insert(0, {matcha!r}); "
        "from cosyvoice.cli.cosyvoice import AutoModel; "
        "print('AutoModel import OK')"
    )
    # Use _run(check=False) so the failure path goes through our standard
    # tail-of-output printer, then raise ourselves so the user still gets
    # a clear "this is a sanity-check failure" error.
    r = _run([py, "-c", code], check=False)
    print(r.stdout, end="")
    if r.returncode != 0:
        _log(
            "8/8 sanity",
            "FAILED. Common causes:",
        )
        _log("8/8 sanity", "  - venv pip install didn't complete (re-run Stage 7)")
        _log("8/8 sanity", "  - missing Matcha-TTS submodule (re-run Stage 5)")
        _log("8/8 sanity", "  - disk full (check Drive quota)")
        raise RuntimeError("AutoModel import failed; see hints above.")

    # Write the discovery JSON so pipeline.py can find this venv without
    # running colab_setup.py first.
    venv_dir = os.environ["COSYV_VENV_DIR"]
    json_path = os.path.join(venv_dir, COSYV_VENV_PATH_JSON)
    import json as _json

    Path(json_path).write_text(
        _json.dumps(
            {
                "venv_dir": venv_dir,
                "venv_python": py,
                "model_dir": os.environ["COSYVOICE_MODEL_DIR"],
                "cosyvoice_repo": repo,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _log("8/8 sanity", f"wrote discovery file {json_path}")
    _log("8/8 sanity", "OK")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YouDub Colab setup (single source of truth).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--drive-root", default=None, help="Override the YouDub project root."
    )
    p.add_argument(
        "--ollama-host",
        default=None,
        help="Set OLLAMA_HOST (point to remote box to skip local install).",
    )
    p.add_argument(
        "--hf-model-id",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face repo id for the CosyVoice3 weights.",
    )
    p.add_argument(
        "--cosyvoice-branch",
        default=DEFAULT_COSYVOICE_BRANCH,
        help="Git branch of FunAudioLLM/CosyVoice to checkout.",
    )
    p.add_argument(
        "--skip-venv",
        action="store_true",
        help="Skip Python 3.11 venv creation/refresh (Stage 2).",
    )
    p.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Skip apt-get install (ffmpeg, sox, git-lfs).",
    )
    p.add_argument(
        "--skip-pip", action="store_true", help="Skip pip install of light deps."
    )
    p.add_argument(
        "--skip-clone", action="store_true", help="Skip CosyVoice git clone/pull."
    )
    p.add_argument(
        "--skip-weights", action="store_true", help="Skip HF weights download."
    )
    p.add_argument(
        "--skip-cosyv-deps",
        action="store_true",
        help="Skip pip install of CosyVoice's pinned requirements "
        "(and the AutoModel sanity check).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    print("=" * 60)
    print("YouDub Colab setup")
    print("=" * 60)

    env = stage_paths(args)
    venv_python = stage_create_cosy_venv(args)
    # Update COSYV_VENV_PYTHON in the env file now that the venv exists.
    env["COSYV_VENV_PYTHON"] = venv_python
    env_sh_path = Path(__file__).resolve().parent / "colab_env.sh"
    lines = [
        "#!/usr/bin/env bash",
        "# Auto-generated by scripts/colab_setup.py — do not edit by hand.",
        "# Re-run colab_setup.py to refresh.",
    ]
    for k, v in env.items():
        lines.append(f'export {k}="{v}"')
    env_sh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stage_system_deps(args)
    stage_pip_light(args)
    repo = stage_clone_cosyvoice(args)
    stage_download_weights(args)
    stage_install_cosyv_deps(args, repo)
    stage_sanity_check(args)

    # Final summary.
    print()
    print("=" * 60)
    print("Setup complete. Env vars in effect for this process:")
    for k, v in env.items():
        print(f"  {k}={v}")
    print()
    print("To make these env vars visible to subsequent Colab cells, run:")
    print(f"    !source {env_sh_path}")
    print()
    print("Two-environment architecture is in place:")
    print(
        f"  - Main Colab kernel (Py{sys.version_info.major}.{sys.version_info.minor}):"
    )
    print(f"      runs the pipeline orchestration + faster-whisper + Ollama + ffmpeg")
    print(f"  - CosyVoice venv (Py{PY311_VERSION}): {venv_python}")
    print(f"      runs the TTS model via scripts/{COSYV_INFER_SCRIPT_NAME}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
