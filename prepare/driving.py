"""Driving (PhysicalAI-AV) in the release format (DATA.md): the front wide camera at 640x360 and 20 fps, and per frame
the action [speed m/s, curvature 1/m] of the online egomotion, as the paper's world models and IDM probes read them.

    python prepare/driving.py --src <PhysicalAI-Autonomous-Vehicles> --out <root> [--split test] [--limit N]

--src is the Hugging Face release as it downloads (a key is a clip id; only the paper's clips, those of the split lists,
are converted):

    clip_index.parquet                                          clip_id -> chunk
    camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_<chunk>.zip
        <clip_id>.camera_front_wide_120fov.mp4                  1920x1080, 30 fps
        <clip_id>.camera_front_wide_120fov.timestamps.parquet   the capture time of every frame (us)
    labels/egomotion/egomotion.chunk_<chunk>.zip
        <clip_id>.egomotion.parquet                             ~100 Hz: timestamp, orientation qx..qw, vx, vy, ...

Frames: the paper's transcode `ffmpeg -i <mp4> -vf fps=20,scale=-2:360`, run as the same libavfilter graph. fps keeps
the last 30 fps frame whose time rounds to at or before each 20 fps tick (source frame ceil(1.5 j) at frame j), scale
is bicubic: a clip of n source frames has (n - 1) * 2 // 3 + 1 (403 for most). The paper then re-encoded them (crf 23);
here they are encoded once, at crf 18.

Actions: frame k is taken to be at tau_k = t_0 + k * 50 ms, t_0 the capture time of the clip's first frame. Row k is
the egomotion interpolated linearly at tau_k (held at its ends): the forward speed, vx and vy rotated from the clip's
world frame into the car's by the yaw of the orientation, and the curvature. The paper's loader read row round(20 t)
for frame floor(20 t), one row ahead on about half of its windows; here row j is the action at frame j."""

import io
import zipfile
from itertools import chain
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from common import arguments, run, write_episode

VIEW = "camera_front_wide_120fov"
FRAME_US = 50_000  # one 20 fps step


def sources(src: Path) -> dict[str, tuple[Path, Path]]:
    """{clip_id: (camera zip, egomotion zip)} of the clips whose two zips are in the download."""
    index = pq.read_table(src / "clip_index.parquet", columns=["clip_id", "chunk"]).to_pydict()
    zips = {
        chunk: (
            src / "camera" / VIEW / f"{VIEW}.chunk_{chunk:04d}.zip",
            src / "labels" / "egomotion" / f"egomotion.chunk_{chunk:04d}.zip",
        )
        for chunk in set(index["chunk"])
    }
    zips = {chunk: pair for chunk, pair in zips.items() if pair[0].exists() and pair[1].exists()}
    return {clip: zips[chunk] for clip, chunk in zip(index["clip_id"], index["chunk"], strict=True) if chunk in zips}


def frames_360p20(mp4: bytes):
    """The mp4's frames through the paper's `fps=20,scale=-2:360` filter graph -> 640x360 av.VideoFrames."""
    with av.open(io.BytesIO(mp4)) as container:
        stream = container.streams.video[0]
        graph = av.filter.Graph()
        filters = graph.add("fps", "20"), graph.add("scale", "-2:360:flags=bicubic")
        graph.link_nodes(graph.add_buffer(template=stream), *filters, graph.add("buffersink")).configure()
        for frame in chain(container.decode(stream), [None]):  # None flushes the graph
            graph.push(frame)
            while True:
                try:
                    yield graph.pull()
                except (av.BlockingIOError, av.EOFError):
                    break


def ego_actions(first_us: int, egomotion: bytes, count: int) -> np.ndarray:
    """(count, 2) float32 [body-frame speed, curvature] of an egomotion parquet at tau_k = first_us + k * 50 ms."""
    table = pq.read_table(io.BytesIO(egomotion))
    t, vx, vy, curvature, qx, qy, qz, qw = (
        np.asarray(table.column(c), dtype=np.float64)
        for c in ("timestamp", "vx", "vy", "curvature", "qx", "qy", "qz", "qw")
    )
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    speed = np.cos(yaw) * vx + np.sin(yaw) * vy
    tau = first_us + FRAME_US * np.arange(count, dtype=np.float64)
    return np.stack([np.interp(tau, t, speed), np.interp(tau, t, curvature)], axis=1).astype(np.float32)


def convert(clip: str, zips: tuple[Path, Path], count: int, out: Path) -> None:
    with zipfile.ZipFile(zips[0]) as camera:
        mp4 = camera.read(f"{clip}.{VIEW}.mp4")
        stamps = pq.read_table(io.BytesIO(camera.read(f"{clip}.{VIEW}.timestamps.parquet")), columns=["timestamp"])
    with zipfile.ZipFile(zips[1]) as labels:
        egomotion = labels.read(f"{clip}.egomotion.parquet")
    actions = ego_actions(stamps.column("timestamp")[0].as_py(), egomotion, count)
    write_episode(out, clip, frames_360p20(mp4), actions, count)


if __name__ == "__main__":
    args = arguments(__doc__).parse_args()
    run("driving", args, sources(args.src), convert)
