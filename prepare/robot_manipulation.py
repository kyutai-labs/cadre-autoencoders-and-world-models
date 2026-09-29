"""robot_manipulation (ABC): XDOF/ABC-130k episodes -> the release format (DATA.md).

    python prepare/robot_manipulation.py --src <ABC-130k> --out <root> [--split S] [--limit N] [--workers N]

Reads `<src>/data/train/<task>/<episode>/episode.mcap` of the RealSense stations (a /top-camera stream, 640x480, no
B-frames); the key is `<task>/<episode>`. As the paper's LeRobot conversion of these files did:
- the 30 fps timeline is the /top-camera messages in log-time order: source frame i is the i-th message;
- the state s_i is [left 6 joints, left gripper, right 6 joints, right gripper], float32: the `position` of
  /left-arm-state, /left-ee-state, /right-arm-state, /right-ee-state, each from the sample whose log_time is nearest
  frame i's (the earlier one on a tie);
- the trailing frames where a state or the matching /*-action command holds a NaN are dropped, leaving n frames.
Then, as the paper's loader read them at the frame time t = j / 20 (its rules at t just after j / 20, which settles
the ties):
- frame j is source frame floor(30 t) = floor(3 j / 2), stretched to 288x512 by the loader's antialiased bilinear
  resize (so dataset.py reads it as stored); the episode has n * 20 // 30 frames;
- actions[j] = s[r(j + 1)] - s[r(j)] with r(j) = round(30 t) = (3 j + 1) // 2, clipped to the episode: the
  joint-state change over one 20 fps step. On odd j the rows lead the frames by one source frame, as on half the
  steps of every paper clip.

--reencode (the default) reproduces a BUG of our data pipeline: our LeRobot conversion re-encoded the top camera
(FFmpeg's bicubic nv12 -> RGB, then libx264 ultrafast, GOP 2, crf 30) before anything read it, so the released
robot-manipulation models were trained and evaluated on those degraded frames. The frames go through the same
re-encode, bit-exact, before the selection. --no-reencode keeps the MCAP's own, cleaner frames, which score the
released models worse for the wrong reason: on 128 paper clips, the released 2x32x32 world model's FID@96 is 42.8 on
the paper's frames, 45.0 with --reencode and 54.8 without; LPIPS 0.146 / 0.145 / 0.162; the probe's real-video
ceiling 0.919 / 0.920 / 0.906. Keep it on to reproduce the paper or use the released weights; turn it off only to
train new models on the original frames.
"""

import argparse
import functools
from fractions import Fraction
from itertools import chain
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from common import FPS, arguments, run, write_episode
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

TOP = "/top-camera"
STATE = ["/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state"]
COMMAND = [topic.replace("state", "action") for topic in STATE]
SOURCE_FPS = 30
SIZE = (288, 512)


def sources(src: Path) -> dict[str, Path]:
    """{<task>/<episode>: its episode.mcap} of the download."""
    paths = sorted((src / "data" / "train").glob("*/episode_*/episode.mcap"))
    return {f"{p.parent.parent.name}/{p.parent.name}": p for p in paths}


def read_mcap(path: Path) -> tuple[np.ndarray, list[bytes], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """The top camera's log times and H.264 packets, and {topic: (log times, float32 positions)} of the others."""
    times, packets, series = [], [], {topic: ([], []) for topic in STATE + COMMAND}
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages(topics=[TOP, *STATE, *COMMAND]):
            if channel.topic == TOP:
                times.append(message.log_time)
                packets.append(decoded.data)
            else:
                series[channel.topic][0].append(message.log_time)
                series[channel.topic][1].append(np.asarray(decoded.position, np.float32))
    return np.array(times), packets, {k: (np.array(t), np.array(v).reshape(len(t), -1)) for k, (t, v) in series.items()}


def nearest(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the sample nearest each query time, the earlier one on a tie."""
    i = np.clip(np.searchsorted(times, query), 1, len(times) - 1)
    return np.where(np.abs(query - times[i - 1]) <= np.abs(query - times[i]), i - 1, i)


def sample(series: dict, topics: list[str], query: np.ndarray) -> np.ndarray:
    """(len(query), 14) float32: each topic's position (an arm's 6 joints, a gripper's position[0]) at its sample
    nearest each query time."""
    columns = [series[t][1][nearest(series[t][0], query), : 6 if "arm" in t else 1] for t in topics]
    return np.concatenate(columns, axis=1)


def states(times: np.ndarray, series: dict) -> np.ndarray:
    """(n, 14) states of the top-camera frames, the frames with a trailing NaN state or command dropped."""
    state = sample(series, STATE, times)
    valid = ~(np.isnan(state).any(1) | np.isnan(sample(series, COMMAND, times)).any(1))
    n = len(valid) - np.argmax(valid[::-1])
    assert valid[:n].all(), "a NaN inside the episode: the paper dropped such episodes"
    return state[:n]


def actions(state: np.ndarray) -> np.ndarray:
    """(n * 20 // 30, 14) deltas of the states over each 20 fps step."""
    rows = np.minimum((3 * np.arange(len(state) * FPS // SOURCE_FPS + 1) + 1) // 2, len(state) - 1)
    return state[rows[1:]] - state[rows[:-1]]


def decoded(packets: list[bytes]):
    """The top camera's frames, one per message."""
    codec = av.CodecContext.create("h264", "r")
    for packet in [*map(av.Packet, packets), None]:  # a message is one access unit
        yield from codec.decode(packet)


def reencoded(frames):
    """The frames as our LeRobot conversion re-encoded them (see --reencode), decoded again."""
    graph = av.filter.Graph()  # the RGB conversion of its FFmpeg decode of NVDEC's nv12 output
    source = graph.add_buffer(width=640, height=480, format="nv12", time_base=Fraction(1, SOURCE_FPS))
    graph.link_nodes(source, graph.add("scale", "flags=bicubic"), graph.add("format", "rgb24"), graph.add("buffersink"))
    graph.configure()
    encoder, decoder = av.CodecContext.create("libx264", "w"), av.CodecContext.create("h264", "r")
    encoder.width, encoder.height, encoder.pix_fmt, encoder.time_base = 640, 480, "yuv420p", Fraction(1, SOURCE_FPS)
    encoder.options = {"preset": "ultrafast", "g": "2", "crf": "30", "threads": "1"}
    for i, frame in enumerate(chain(frames, [None])):
        if frame is not None:
            graph.push(frame.reformat(format="nv12"))
            frame = av.VideoFrame.from_ndarray(graph.vpull().to_ndarray(), format="rgb24")
            frame.pts = i
        for packet in encoder.encode(frame):
            yield from decoder.decode(packet)
    yield from decoder.decode(None)


def frames(packets: list[bytes], count: int, reencode: bool):
    """The count 20 fps frames, source frame floor(3 j / 2) for frame j, stretched to 288x512."""
    keep, shown = {3 * j // 2 for j in range(count)}, 0
    for shown, frame in enumerate(reencoded(decoded(packets)) if reencode else decoded(packets), 1):
        if shown - 1 in keep:
            image = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)[None].float()
            image = F.interpolate(image, SIZE, mode="bilinear", antialias=True).round().clamp(0, 255)
            yield av.VideoFrame.from_ndarray(image[0].permute(1, 2, 0).to(torch.uint8).numpy(), format="rgb24")
    assert shown == len(packets), f"{shown} frames decoded from {len(packets)} top-camera messages"


def convert(key: str, path: Path, count: int, out: Path, reencode: bool = True) -> None:
    torch.set_num_threads(1)  # the episodes run in parallel processes
    times, packets, series = read_mcap(path)
    rows = actions(states(times, series))
    write_episode(out, key, frames(packets, len(rows), reencode), rows, count)


if __name__ == "__main__":
    parser = arguments(__doc__)
    parser.add_argument(
        "--reencode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reproduce our pipeline's lossy re-encode (a bug) that the released models were trained and evaluated on "
        "(default: on, as the paper; off only to train new models on the original frames; see above)",
    )
    args = parser.parse_args()
    run("robot_manipulation", args, sources(args.src), functools.partial(convert, reencode=args.reencode))
