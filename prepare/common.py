"""What the four prepare scripts share: the command line, the paper's split lists (manifests/splits/<env>/), the
episode writer (H.264 at 20 fps, then the actions) and the resumable loop over episodes in parallel."""

import argparse
import multiprocessing
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

FPS = 20
SPLITS = Path(__file__).resolve().parent.parent / "manifests" / "splits"


EPILOG = """The defaults reproduce our preprocessing exactly: change no option to match the paper. Run in the pinned
environment (uv sync --extra prepare). Downloads, commands, times and disk space: DATA.md, "Preparing the data"."""


def arguments(doc: str) -> argparse.ArgumentParser:
    """The options of every prepare script; a script adds its own before parsing."""
    formatter = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=doc, epilog=EPILOG, formatter_class=formatter)
    parser.add_argument("--src", type=Path, required=True, help="the downloaded release, as it unpacks")
    parser.add_argument("--out", type=Path, required=True, help="the dataset root to write (DATA.md)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, help="convert only the split's first N episodes in --src")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def read_split(env: str, split: str) -> dict[str, int]:
    """{key: frames} of the paper's split list."""
    words = (SPLITS / env / f"{split}.txt").read_text().split()
    return dict(zip(words[::2], map(int, words[1::2]), strict=True))


def write_episode(out: Path, key: str, frames: Iterable[av.VideoFrame], actions: np.ndarray, count: int) -> None:
    """Writes videos/<key>.mp4 (libx264, crf 18, yuv420p, 20 fps: frame j at j / 20 s) from the frames as given,
    then actions/<key>.npy, whose presence marks the episode finished. Both must have `count` rows, the frame count
    of the paper's split list."""
    video, rows = out / "videos" / f"{key}.mp4", out / "actions" / f"{key}.npy"
    video.parent.mkdir(parents=True, exist_ok=True)
    rows.parent.mkdir(parents=True, exist_ok=True)
    partial = video.with_name(video.stem + ".partial.mp4")
    with av.open(str(partial), "w") as container:
        stream = container.add_stream("libx264", rate=FPS, options={"crf": "18"})
        for j, frame in enumerate(frames):
            if j == 0:
                stream.width, stream.height, stream.pix_fmt = frame.width, frame.height, "yuv420p"
            frame.pts, frame.time_base = j, Fraction(1, FPS)
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    assert j + 1 == len(actions) == count, f"{key}: {j + 1} frames, {len(actions)} action rows, split lists {count}"
    os.replace(partial, video)
    with open(rows.with_name(rows.stem + ".partial.npy"), "wb") as f:
        np.save(f, np.asarray(actions, dtype=np.float32))
    os.replace(f.name, rows)


def run(env: str, args: argparse.Namespace, sources: dict, convert: Callable, chunksize: int = 1) -> None:
    """Converts the episodes of --split that the download holds (sources: key -> where it is in --src) with
    convert(key, source, count, out), count the episode's frames in the paper's split list, in --workers spawned
    processes, skipping finished episodes; then writes <out>/splits/ for every finished episode."""
    splits = {split: read_split(env, split) for split in ("train", "val", "test")}
    chosen = {key for split, listed in splits.items() if args.split in ("all", split) for key in listed}
    keys = [key for key in sources if key in chosen][: args.limit]  # in the download's order (e.g. shard by shard)
    todo = [key for key in keys if not (args.out / "actions" / f"{key}.npy").exists()]
    frames = {key: n for split in splits.values() for key, n in split.items()}
    print(f"{env}: {len(keys)} episodes of split {args.split} in {args.src}, {len(todo)} to convert", flush=True)
    jobs = [sources[k] for k in todo], [frames[k] for k in todo], [args.out] * len(todo)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, (key, _) in enumerate(zip(todo, pool.map(convert, todo, *jobs, chunksize=chunksize), strict=True), 1):
            print(f"[{i}/{len(todo)}] {key}: {frames[key]} frames", flush=True)
    (args.out / "splits").mkdir(parents=True, exist_ok=True)
    for split, listed in splits.items():
        done = [f"{key} {n}\n" for key, n in listed.items() if (args.out / "actions" / f"{key}.npy").exists()]
        (args.out / "splits" / f"{split}.txt").write_text("".join(done))
