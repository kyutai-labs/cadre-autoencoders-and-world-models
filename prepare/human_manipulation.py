"""Human manipulation (EgoDex): Apple's part zips -> the release format (DATA.md), as the paper's world models and IDM
probes read it.

    python prepare/human_manipulation.py --src <dir of part1.zip .. part5.zip, extra.zip> --out <root> \\
        --mano_dir <mano_v1_2/models>

Source: <src>/<part>.zip holding <part>/<task>/<index>.mp4 (1920x1080, 30 fps) and .hdf5 (one row per video frame);
the key is <part>/<task>/<index>. The official test.zip was never used. --mano_dir holds MANO_LEFT.pkl and
MANO_RIGHT.pkl of MANO v1.2 (https://mano.is.tue.mpg.de, non-commercial licence); chumpy is not needed.

Video: 20 fps frame j is 30 fps frame floor(3 j / 2), the one on screen at j / 20 s, scaled by the paper's transcode
filter (FFmpeg `scale=512:288:flags=lanczos`), so the dataset reads it unresized.

Actions (102, 51 per hand, left first): row j is the step from the pose at 30 fps frame floor(3 j / 2) to the pose at
floor(3 (j + 1) / 2) (the last clamped to the episode's last frame, as the paper loader did): the translation
difference, and for the wrist and each of the 15 finger joints the axis-angle of R_a^T R_b. A hand's pose is the
/transforms/<side>Hand translation and rotation (as float32 axis-angle), then the 45 finger-joint axis-angles of a
MANO hand (flat_hand_mean=False: its mean pose added) fitted frame by frame to 21 keypoints of the HDF5 (wrist,
knuckle, intermediate base and tip, and tip of each finger) by 250 Adam steps (lr 0.04, shape regularized by 0.01)
from the rest pose at the wrist. The paper fitted pools of 16384 frames; each frame's parameters see only its own
loss, but Adam's eps makes the fit depend on the loss scale, so the loss is scaled as in those pools. A GPU is used
when there is one.
"""

import functools
import io
import pickle
import zipfile
from pathlib import Path

import av
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from common import arguments, run, write_episode
from scipy.spatial.transform import Rotation

POOL = 16384  # frames per batch of the paper's fit, which set its loss scale
TIPS = [744, 320, 443, 554, 671]  # MANO's thumb, index, middle, ring and pinky tip vertices
JOINTS = ("Knuckle", "IntermediateBase", "IntermediateTip")
KEYPOINTS = [  # the fitted transforms of a hand (after "left" or "right"), in the order of MANO's 16 joints and 5 tips
    "Hand",
    *(f"{finger}Finger{joint}" for finger in ("Index", "Middle", "Little", "Ring") for joint in JOINTS),
    *(f"Thumb{joint}" for joint in (*JOINTS, "Tip")),
    *(f"{finger}FingerTip" for finger in ("Index", "Middle", "Ring", "Little")),
]


class _Chumpy:  # stands in for the chumpy arrays of the MANO pickles
    def __setstate__(self, state):
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):  # MANO's pickles also name scipy.sparse.csc, deprecated in SciPy
    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return _Chumpy
        return super().find_class("scipy.sparse" if module.startswith("scipy.sparse") else module, name)


@functools.cache
def mano(mano_dir: str, side: str) -> dict:
    """The MANO buffers the fit needs, as float32 tensors on the GPU if there is one."""
    with open(Path(mano_dir) / f"MANO_{side.upper()}.pkl", "rb") as f:
        m = _Unpickler(f, encoding="latin1").load()
    select = m["shapedirs"].__dict__  # a chumpy Select of a (778, 3, 20) array
    shapedirs = select["a"].__dict__["x"].ravel()[select["idxs"]].reshape(select["preferred_shape"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = functools.partial(torch.tensor, dtype=torch.float32, device=device)
    tips = torch.tensor(TIPS, device=device)
    posedirs = t(np.reshape(m["posedirs"], [-1, 135]).T)
    return {
        "v_template": t(m["v_template"]),
        "shapedirs": t(shapedirs),
        "J_regressor": t(m["J_regressor"].toarray()),
        "pose_mean": t(np.concatenate([np.zeros(3), m["hands_mean"]])),
        "parents": m["kintree_table"][0].tolist(),
        "tip_template": t(m["v_template"][TIPS]),
        "tip_shapedirs": t(shapedirs[TIPS]),
        "tip_weights": t(m["weights"][TIPS]),
        "tip_posedirs": posedirs.view(135, 778, 3).index_select(1, tips).reshape(135, 15).T,
    }


def rodrigues(r: torch.Tensor) -> torch.Tensor:
    """(N, 3) axis-angles -> (N, 3, 3) rotations, as MANO's batch_rodrigues."""
    angle = torch.norm(r + 1e-8, dim=1, keepdim=True)
    x, y, z = torch.split(r / angle, 1, dim=1)
    zero = torch.zeros_like(x)
    K = torch.cat([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).view(-1, 3, 3)
    return torch.eye(3, device=r.device) + torch.sin(angle)[:, None] * K + (1 - torch.cos(angle))[:, None] * (K @ K)


def mano_keypoints(m: dict, beta, orient, theta, transl) -> torch.Tensor:
    """MANO's 16 posed joints and 5 fingertips (N, 21, 3), skinning only the 5 tip vertices."""
    n = len(beta)
    rot = rodrigues((torch.cat([orient, theta], dim=1) + m["pose_mean"]).view(-1, 3)).view(n, 16, 3, 3)
    joints = torch.einsum(
        "bik,ji->bjk", m["v_template"] + torch.einsum("bl,mkl->bmk", beta, m["shapedirs"]), m["J_regressor"]
    ).unsqueeze(-1)
    rel = joints.clone()
    rel[:, 1:] -= joints[:, m["parents"][1:]]
    local = torch.cat([F.pad(rot, [0, 0, 0, 1]), F.pad(rel, [0, 0, 0, 1], value=1)], dim=3)
    chain = [local[:, 0]]
    for i in range(1, 16):
        chain.append(chain[m["parents"][i]] @ local[:, i])
    world = torch.stack(chain, dim=1)
    skin = world - F.pad(world @ F.pad(joints, [0, 0, 0, 1]), [3, 0, 0, 0, 0, 0, 0, 0])
    posed = (rot[:, 1:] - torch.eye(3, device=rot.device)).reshape(n, -1) @ m["tip_posedirs"].T
    tips = m["tip_template"] + torch.einsum("bl,mkl->bmk", beta, m["tip_shapedirs"]) + posed.view(n, 5, 3)
    tip_skin = (m["tip_weights"].expand(n, -1, -1) @ skin.view(n, 16, 16)).view(n, 5, 4, 4)
    tips = (tip_skin @ torch.cat([tips, torch.ones_like(tips[..., :1])], dim=2).unsqueeze(-1))[:, :, :3, 0]
    return torch.cat([world[:, :, :3, 3], tips], dim=1) + transl.unsqueeze(1)


def fit_fingers(points: torch.Tensor, m: dict) -> np.ndarray:
    """(N, 21, 3) keypoints -> (N, 45) float32 finger-joint axis-angles of the fitted MANO hand, mean pose added."""
    theta, beta, orient = (torch.zeros(len(points), k, device=points.device, requires_grad=True) for k in (45, 10, 3))
    transl = points[:, 0].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([theta, beta, orient, transl], lr=0.04)
    for _ in range(250):
        error = ((mano_keypoints(m, beta, orient, theta, transl) - points) ** 2).sum(-1)
        loss = error.sum() / (21 * POOL) + 0.01 * (beta**2).sum() / POOL
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return (theta.detach() + m["pose_mean"][3:]).cpu().numpy()


def hand_poses(transforms: h5py.Group, side: str, mano_dir: str) -> np.ndarray:
    """(frames, 51) float32 poses of one hand: wrist translation, wrist axis-angle, 15 finger-joint axis-angles."""
    wrist = np.asarray(transforms[f"{side}Hand"])
    points = np.stack([np.asarray(transforms[side + name])[:, :3, 3] for name in KEYPOINTS], axis=1)
    m = mano(mano_dir, side)
    fingers = fit_fingers(torch.from_numpy(points).to(m["v_template"].device), m)
    return np.concatenate(
        [wrist[:, :3, 3], Rotation.from_matrix(wrist[:, :3, :3]).as_rotvec().astype(np.float32), fingers], axis=1
    )


def steps(poses: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(frames, 102) poses -> (len(rows) - 1, 102) steps between consecutive rows."""
    a, b = poses[rows[:-1]], poses[rows[1:]]
    out = b - a
    for h in (0, 51):
        ra, rb = (Rotation.from_rotvec(p[:, h + 3 : h + 51].reshape(-1, 3).astype(np.float64)) for p in (a, b))
        out[:, h + 3 : h + 51] = (ra.inv() * rb).as_rotvec().reshape(len(a), 48)
    return out


def frames(mp4: bytes, shown: set[int]):
    """The 30 fps frames of the mp4 whose index is in `shown`, scaled to 512x288 by FFmpeg's Lanczos."""
    with av.open(io.BytesIO(mp4)) as container:
        stream = container.streams.video[0]
        graph = av.filter.Graph()
        graph.link_nodes(
            graph.add_buffer(template=stream), graph.add("scale", "512:288:flags=lanczos"), graph.add("buffersink")
        ).configure()
        for i, frame in enumerate(container.decode(stream)):
            if i in shown:
                graph.push(frame)
                yield graph.vpull()


@functools.cache
def part(archive: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(archive)  # opened once per worker: reading a part's directory takes 0.5 s


def convert(key: str, archive: str, count: int, out: Path, mano_dir: str) -> None:
    torch.set_num_threads(1)  # the episodes run in parallel processes
    with h5py.File(io.BytesIO(part(archive).read(f"{key}.hdf5")), "r") as h5:
        poses = np.concatenate([hand_poses(h5["transforms"], side, mano_dir) for side in ("left", "right")], axis=1)
    rows = np.minimum(3 * np.arange(count + 1) // 2, len(poses) - 1)
    video = frames(part(archive).read(f"{key}.mp4"), set(rows[:-1].tolist()))
    write_episode(out, key, video, steps(poses, rows), count)


def sources(src: Path) -> dict[str, str]:
    """{key: part zip} of every episode of the part zips in src."""
    return {
        name.removesuffix(".mp4"): str(archive)
        for archive in sorted(src.glob("*.zip"))
        for name in zipfile.ZipFile(archive).namelist()
        if name.endswith(".mp4")
    }


if __name__ == "__main__":
    parser = arguments(__doc__)
    parser.add_argument(
        "--mano_dir",
        type=Path,
        required=True,
        help="the folder holding MANO_LEFT.pkl and MANO_RIGHT.pkl: mano_v1_2/models of MANO v1.2's Models & Code "
        "download (https://mano.is.tue.mpg.de, DATA.md)",
    )
    args = parser.parse_args()
    run("human_manipulation", args, sources(args.src), functools.partial(convert, mano_dir=str(args.mano_dir)))
