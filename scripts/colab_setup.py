"""
YouDub Colab setup script.

This is the SINGLE source of truth for Colab environment setup. It replaces
the previous 11 inline notebook cells so that fixes only need to happen
in this repo (not in the Colab notebook). Re-running the script is safe;
each stage has its own idempotency check.

Usage from a Colab cell:

    !python "{DRIVE_ROOT}/scripts/colab_setup.py"
    !source  "{DRIVE_ROOT}/scripts/colab_env.sh"   # export env vars to this cell

CLI flags (all default to "do the thing"; flags are for power users):

    --drive-root PATH    Override the YouDub project root (default: $DRIVE_ROOT
                         from env, or /content/drive/MyDrive/YouDub).
    --ollama-host URL    Set OLLAMA_HOST (default: http://localhost:11434).
    --hf-model-id ID     HF repo to download CosyVoice3 weights from.
    --skip-system-deps   Skip apt-get install.
    --skip-pip           Skip pip install of light deps.
    --skip-clone         Skip CosyVoice git clone/pull.
    --skip-weights       Skip HF weights download.
    --skip-cosyv-deps    Skip pip install of CosyVoice's pinned requirements.
    --cosyvoice-branch   Git branch/tag to checkout (default: main).

Stages
------
1. Resolve & export paths (DRIVE_ROOT, COSYVOICE_REPO_DIR, COSYVOICE_MODEL_DIR, OLLAMA_HOST).
2. System deps (ffmpeg, sox, git-lfs).
3. Light Python deps (yt-dlp, faster-whisper, ollama, etc.).
4. Clone or update FunAudioLLM/CosyVoice (with submodules).
5. Snapshot-download CosyVoice3 weights from Hugging Face.
6. Install CosyVoice's pinned requirements.
7. Sanity-check: import AutoModel from the cloned CosyVoice repo.

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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _log(stage: str, msg: str) -> None:
    """One-line stage-prefixed log."""
    print(f"[{stage}] {msg}", flush=True)


def _run(
    cmd: list[str], *, cwd: Optional[str] = None, check: bool = True
) -> subprocess.CompletedProcess:
    """subprocess.run with a printed command, captured stderr on failure."""
    pretty = " ".join(cmd) if isinstance(cmd, list) else cmd
    if cwd:
        pretty = f"(cwd={cwd}) {pretty}"
    print(f"  $ {pretty}", flush=True)
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


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
    ollama_host = (
        args.ollama_host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    )

    env_vars = {
        "DRIVE_ROOT": drive_root,
        "COSYVOICE_REPO_DIR": cosy_repo,
        "COSYVOICE_MODEL_DIR": cosy_model,
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
    _log("1/7 paths", f"DRIVE_ROOT={drive_root}")
    _log("1/7 paths", f"COSYVOICE_REPO_DIR={cosy_repo}")
    _log("1/7 paths", f"COSYVOICE_MODEL_DIR={cosy_model}")
    _log("1/7 paths", f"OLLAMA_HOST={ollama_host}")
    _log("1/7 paths", f"wrote {env_sh_path}")
    return env_vars


# ---------------------------------------------------------------------------
# Stage 2: system deps
# ---------------------------------------------------------------------------


def stage_system_deps(args: argparse.Namespace) -> None:
    if args.skip_system_deps:
        _log("2/7 system-deps", "skipped (--skip-system-deps)")
        return
    if all(_which(p) for p in ["ffmpeg", "sox", "git-lfs"]):
        _log("2/7 system-deps", "ffmpeg, sox, git-lfs already present; skipping")
        return
    _log("2/7 system-deps", "apt-get update + install ffmpeg, sox, libsox-dev, git-lfs")
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
    _log("2/7 system-deps", out.stdout.splitlines()[0])


# ---------------------------------------------------------------------------
# Stage 3: light Python deps
# ---------------------------------------------------------------------------


def stage_pip_light(args: argparse.Namespace) -> None:
    if args.skip_pip:
        _log("3/7 pip-light", "skipped (--skip-pip)")
        return
    missing = [
        p for p in LIGHT_PIP_DEPS if not _pip_installed(p.split(">=")[0].split("==")[0])
    ]
    if not missing:
        _log("3/7 pip-light", "all light deps already installed; skipping")
        return
    _log("3/7 pip-light", f"installing {len(missing)} missing deps")
    _run([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])


# ---------------------------------------------------------------------------
# Stage 4: clone or update CosyVoice
# ---------------------------------------------------------------------------


def stage_clone_cosyvoice(args: argparse.Namespace) -> str:
    if args.skip_clone:
        _log("4/7 clone", "skipped (--skip-clone)")
        return os.environ["COSYVOICE_REPO_DIR"]
    repo = os.environ["COSYVOICE_REPO_DIR"]
    if not os.path.isdir(repo):
        _log(
            "4/7 clone",
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
        _log("4/7 clone", f"{repo} already exists; pulling + updating submodules")
        # Don't blow up on a failed pull (e.g. transient network) — the repo
        # is still usable if the previous clone succeeded.
        try:
            _run(["git", "-C", repo, "pull", "--recurse-submodules"], check=False)
        except Exception as e:
            _log("4/7 clone", f"pull failed (non-fatal): {e}")
    # Always run submodule update to recover from a partial previous clone.
    _run(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
    matcha = os.path.join(repo, "third_party", "Matcha-TTS")
    if not os.path.isdir(matcha):
        raise RuntimeError(
            f"Matcha-TTS submodule missing at {matcha}. Check the clone step above."
        )
    _log("4/7 clone", "Matcha-TTS present")
    return repo


# ---------------------------------------------------------------------------
# Stage 5: download CosyVoice3 weights
# ---------------------------------------------------------------------------


def stage_download_weights(args: argparse.Namespace) -> str:
    if args.skip_weights:
        _log("5/7 weights", "skipped (--skip-weights)")
        return os.environ["COSYVOICE_MODEL_DIR"]
    model_dir = os.environ["COSYVOICE_MODEL_DIR"]
    marker = os.path.join(model_dir, "cosyvoice3.yaml")
    if os.path.isdir(model_dir) and os.path.isfile(marker):
        _log("5/7 weights", f"cosyvoice3.yaml already present at {model_dir}; skipping")
        return model_dir
    _log("5/7 weights", f"snapshot_download {args.hf_model_id} -> {model_dir}")
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
    _log("5/7 weights", "download complete")
    # Report size.
    out = subprocess.run(["du", "-sh", model_dir], capture_output=True, text=True)
    if out.stdout.strip():
        _log("5/7 weights", f"on-disk size: {out.stdout.split()[0]}")
    return model_dir


# ---------------------------------------------------------------------------
# Stage 6: install CosyVoice's pinned requirements
# ---------------------------------------------------------------------------

COSYV_DEPS_MARKER = ".youdub_cosyv_deps_installed"


def stage_install_cosyv_deps(args: argparse.Namespace, repo: str) -> None:
    if args.skip_cosyv_deps:
        _log("6/7 cosyv-deps", "skipped (--skip-cosyv-deps)")
        return
    marker = os.path.join(repo, COSYV_DEPS_MARKER)
    if os.path.isfile(marker):
        _log("6/7 cosyv-deps", f"marker {marker} present; skipping")
        return
    req = os.path.join(repo, "requirements.txt")
    if not os.path.isfile(req):
        raise RuntimeError(f"CosyVoice requirements.txt missing at {req}")
    _log("6/7 cosyv-deps", "pip install -r requirements.txt (2-3 min)")
    _run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=repo,
    )
    Path(marker).write_text(
        f"Installed by colab_setup.py on {os.environ.get('HOSTNAME', 'unknown')}\n",
        encoding="utf-8",
    )
    _log("6/7 cosyv-deps", f"wrote marker {marker}")


# ---------------------------------------------------------------------------
# Stage 7: AutoModel import sanity check
# ---------------------------------------------------------------------------


def stage_sanity_check(args: argparse.Namespace) -> None:
    if args.skip_cosyv_deps:
        _log("7/7 sanity", "skipped (--skip-cosyv-deps)")
        return
    repo = os.environ["COSYVOICE_REPO_DIR"]
    matcha = os.path.join(repo, "third_party", "Matcha-TTS")
    _log("7/7 sanity", "importing AutoModel from cloned CosyVoice")
    code = (
        "import sys; "
        f"sys.path.insert(0, {repo!r}); "
        f"sys.path.insert(0, {matcha!r}); "
        "from cosyvoice.cli.cosyvoice import AutoModel; "
        "print('AutoModel import OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, end="")
        print(r.stderr, end="", file=sys.stderr)
        _log(
            "7/7 sanity", "FAILED. Most common cause: onnxruntime/numpy version drift."
        )
        _log("7/7 sanity", "Fix: in the CosyVoice dir, run:")
        _log("7/7 sanity", "  pip install -U onnxruntime numpy==1.26.4")
        raise RuntimeError("AutoModel import failed; see hints above.")
    print(r.stdout, end="")
    _log("7/7 sanity", "OK")


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
    repo = os.environ["COSYVOICE_REPO_DIR"]

    stage_system_deps(args)
    stage_pip_light(args)
    stage_clone_cosyvoice(args)
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
    print(f"    !source {Path(__file__).resolve().parent / 'colab_env.sh'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
