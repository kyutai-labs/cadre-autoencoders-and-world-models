"""Training checkpoints: <output_dir>/checkpoint-<step>/ holding the release checkpoint (checkpoint.pt, the averaged
weights, loadable by the model's from_checkpoint) and the full training state (training_state.pt).

<step> is the last optimizer step taken (0-based), so a run of N steps ends with checkpoint-<N - 1> and a resumed
run starts at <step> + 1. Directories are written under a .tmp name and renamed, so a killed save never leaves a
checkpoint that looks complete.
"""

import re
import shutil
from pathlib import Path

import torch

from util import distributed

RELEASE, TRAINING_STATE = "checkpoint.pt", "training_state.pt"


def checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """The complete checkpoint directories of a run, by step."""
    found = []
    for path in Path(output_dir).glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if m and (path / RELEASE).is_file() and (path / TRAINING_STATE).is_file():
            found.append((int(m.group(1)), path))
    return sorted(found)


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    found = checkpoints(output_dir)
    return found[-1][1] if found else None


def checkpoint_dir(path: str | Path) -> Path:
    """A checkpoint directory, or a run directory (its latest checkpoint)."""
    path = Path(path)
    if (path / TRAINING_STATE).is_file():
        return path
    latest = latest_checkpoint(path)
    assert latest is not None, f"no checkpoint in {path}"
    return latest


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    release: dict,
    training_state: dict,
    keep_recent: int,
    keep_every: int | None,
    final: bool = False,
) -> Path:
    """Write checkpoint-<step>, then prune: checkpoints at multiples of keep_every are permanent, the others are
    kept only while they are among the keep_recent latest, and the final save keeps none of them."""
    output_dir = Path(output_dir)
    target = output_dir / f"checkpoint-{step}"
    staging = output_dir / f"checkpoint-{step}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    torch.save(release, staging / RELEASE)
    torch.save(training_state, staging / TRAINING_STATE)
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)

    def permanent(s: int) -> bool:
        return bool(keep_every) and s % keep_every == 0

    others = [p for s, p in checkpoints(output_dir) if p != target and not permanent(s)]
    keep = 0 if final else keep_recent - (0 if permanent(step) else 1)
    for path in others[: max(0, len(others) - keep)]:
        shutil.rmtree(path)
    return target


def resume_source(
    output_dir: str | Path, continue_from: str | None, finetune_from: str | None
) -> tuple[str, Path] | None:
    """Where a run starts from: ("resume", its own latest checkpoint), ("continue", continue_from),
    ("finetune", finetune_from) or None (fresh). Rank 0 decides, so every rank resumes at the same step."""
    assert not (continue_from and finetune_from), "set at most one of continue_from and finetune_from"
    source = None
    if distributed.is_main():
        if (own := latest_checkpoint(output_dir)) is not None:
            source = ("resume", own)
        elif continue_from:
            source = ("continue", checkpoint_dir(continue_from))
        elif finetune_from:
            source = ("finetune", Path(finetune_from))
    return distributed.broadcast_object(source)
