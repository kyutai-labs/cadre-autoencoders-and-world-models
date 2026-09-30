"""Interactive demo of the CADRE world models: drive them from a browser, at the video's 20 fps.

    uv sync --extra demo
    uv run python demo.py

then open http://localhost:8000 (on a cluster node: ssh -N -L 8000:<node>:8000 <login host>).

One process hosts the world models of one autoencoder, <weights>/<environment> for each environment of --envs (by
default the Hub repo kyutai/cadre-world-models-2x16x16; a local directory of <environment>/<autoencoder>.pt files also
works), sharing a single copy of the autoencoder they nest, and serves one browser session at a time over a websocket (a
new connection takes over). Each environment rolls out from a real prompt: the first 40 frames of a window of its --data
root (DATA.md; the --manifest windows, else the split's; by default the six clips per environment of the weights'
demo_clips/), which also brings the recorded actions of the replay mode. The prompt only primes the world model (its
latents and their KV cache): the page shows the generated frames alone, from the first step on, as the paper's demo did.
A step reads the current action, predicts the next latent with the world model's KV cache (next_latent: denoise_steps
Euler steps, the new frame cached at context noise), decodes its tau frames as the rollout evaluation streams them (a
window of latents sliding one latent a step, the decoder's noise an AR(1) buffer), and sends them as JPEG, paced at 20
fps; the next step runs while they play. All of it under bf16 autocast with fp32 weights, as the rollout evaluation
runs, compiled with CUDA graphs (a few minutes at startup; --no-compile runs eager, about 2.5x slower). The default
denoise_steps is the most that keeps real time on an H100 (DENOISE_STEPS).

Actions are per video frame, in each dataset's units, and stay inside the training range. The keys mirror each other
across the two hands (left: Q W E / A S D, right: U I O / J K L):
- robot_manipulation: 14 joint-state deltas. Per arm, the base joint J0 has its own keys and one more joint (J1-J5
  or the gripper) is selected and moved; a press is 3 standard deviations of a training step's delta.
- driving: [speed m/s, curvature 1/m], integrated from throttle, brake, hard brake and steering inside the training
  envelope. The page draws the commanded path over the video (DrivingControls.ribbon).
- indoor_navigation: the camera twist in the clip's normalized units, moved by the keys and turned by the mouse (or
  the keys); the world model derives its discrete keys from it.
- human_manipulation: 102-D hand-pose deltas. The keys move each wrist along the world axes; its rotation and finger
  joints hold still.
Every environment has a replay mode: the prompt clip's recorded actions, time-aligned with the clip; at their end the
replay moves on to the next window of the environment, as the page's next-clip key does.
"""

import argparse
import asyncio
import json
import math
import queue
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from torchvision.io import encode_jpeg

from dataset import ENVIRONMENTS, FPS, RESOLUTION, EnvironmentDataset
from model_autoencoder import CADRE
from model_world_model import StreamingDecoder, encode_video, load_checkpoint

HERE = Path(__file__).parent
ROLLOUT = yaml.safe_load((HERE / "configs/eval/rollout.yaml").read_text())  # the rollout evaluation's protocol
CONTEXT_FRAMES = ROLLOUT["context_frames"]
CLIP_FRAMES = CONTEXT_FRAMES + ROLLOUT["generated_frames"]
AUTOENCODERS = ("cadre_2x16x16", "cadre_2x32x32")  # cells with world models, the released one (the default) first
MODES = ("interactive", "replay")  # the action sources of every environment, the default first
JPEG_QUALITY = 88
# Default denoising steps: the most that keep real time on one H100, compiled (a step of tau frames within tau / 20 s:
# 92 ms at 10 for 2x32x32, 100 ms at 7 for 2x16x16; the rollout evaluation takes 10).
DENOISE_STEPS = {"cadre_2x32x32": 10, "cadre_2x16x16": 7}
CONTEXT_NOISE = 0.15  # the demo's default; the rollout evaluation re-noises the cached frames at 0.2

TITLES = {
    "robot_manipulation": "Manipulation (Robot)",
    "driving": "Navigation (Driving)",
    "indoor_navigation": "Navigation (Indoor)",
    "human_manipulation": "Manipulation (Human)",
}


class Controls:
    """Keys -> per-frame actions of one environment. Keys are KeyboardEvent.code values ("Shift" for either side).
    `hints` are the UI's (key codes, what they do) pairs."""

    mouse_look = False  # takes mouse motion (look)
    hints: tuple[tuple[tuple[str, ...], str], ...] = ()

    def reset(self, action: np.ndarray) -> None:
        """Take over from the last action the rollout was fed."""

    def press(self, code: str) -> None:
        """Edge-triggered keys: called once per key press."""

    def actions(self, keys: set[str], n: int) -> np.ndarray:
        """The next n frames' actions (n, action_dim) while `keys` are held."""
        raise NotImplementedError

    def hud(self) -> dict:
        return {}


class RobotControls(Controls):
    """ABC's [left J0..J5, left gripper, right J0..J5, right gripper] deltas, held constant over a held key."""

    # A held key's per-frame delta: p97.5-97.8 of the training steps' |delta| of every joint (their p99 is 4.1-4.8),
    # and for an arm joint within the p99.9 of its 1 s mean (3.4-5.1), a held key being a sustained motion.
    press_sigma = 3.0
    joints = ("J1", "J2", "J3", "J4", "J5", "grip")  # selectable, dims 1..6 of each arm; J0 is dim 0
    offset = {"left": 0, "right": 7}
    moves = {  # key -> (arm, J0 or the selected joint, sign); a positive J0 delta turns the base left on screen
        "KeyA": ("left", "base", 1.0),
        "KeyD": ("left", "base", -1.0),
        "KeyW": ("left", "joint", 1.0),
        "KeyS": ("left", "joint", -1.0),
        "KeyJ": ("right", "base", 1.0),
        "KeyL": ("right", "base", -1.0),
        "KeyI": ("right", "joint", 1.0),
        "KeyK": ("right", "joint", -1.0),
    }
    selects = {"KeyQ": ("left", -1), "KeyE": ("left", 1), "KeyU": ("right", -1), "KeyO": ("right", 1)}
    hints = (
        (("KeyQ", "KeyE"), "left select"),
        (("KeyW", "KeyS"), "left joint"),
        (("KeyA", "KeyD"), "left J0"),
        (("KeyU", "KeyO"), "right select"),
        (("KeyI", "KeyK"), "right joint"),
        (("KeyJ", "KeyL"), "right J0"),
    )

    def __init__(self, action_std: list[float]):
        self.press_size = self.press_sigma * np.asarray(action_std, dtype=np.float32)
        self.selected = {"left": 1, "right": 1}
        self.active: list[int] = []

    def select(self, arm: str, joint: int) -> None:
        self.selected[arm] = min(max(int(joint), 1), len(self.joints))

    def press(self, code: str) -> None:
        if code in self.selects:
            arm, step = self.selects[code]
            self.selected[arm] = (self.selected[arm] - 1 + step) % len(self.joints) + 1

    def actions(self, keys: set[str], n: int) -> np.ndarray:
        a = np.zeros(len(self.press_size), dtype=np.float32)
        for code in keys & self.moves.keys():
            arm, joint, sign = self.moves[code]
            dim = self.offset[arm] + (0 if joint == "base" else self.selected[arm])
            a[dim] += sign * self.press_size[dim]
        self.active = np.flatnonzero(a).tolist()
        return np.tile(a, (n, 1))

    def hud(self) -> dict:
        return {"selected": dict(self.selected), "active": self.active}


class DrivingControls(Controls):
    """PhysicalAI-AV's [speed, curvature] as vehicle state, integrated per frame, with the paper demo's dynamics. Speed
    persists when the throttle is released (mild drag) and never reverses; the throttle tapers with speed, braking
    beats it, X brakes hard. The wheel turns at a fixed rate toward full lock and self-centers exponentially once
    released. The bounds come from the training actions: speed up to its 99th percentile, |curvature| up to its 99th
    percentile and within the lateral acceleration v^2 |k| of its 99th percentile, and steering brakes the car toward
    the 95th-percentile speed of its curvature (drivers slow down to turn), which in turn lets the turn tighten.
    Positive curvature steers left."""

    v_max, accel, brake, hard_brake, drag = 29.0, 3.0, 10.0, 25.0, 0.4  # m/s; m/s^2, accel from rest
    k_max, k_rate, k_return, lateral = 0.09, 0.3, 0.4, 2.9  # 1/m, 1/m/s, s (self-centering time constant), m/s^2
    corner_brake = 6.0  # m/s^2, toward the speed of the curvature
    camera_hfov, camera_height, ribbon_seconds = 120.0, 1.6, 3.0  # degrees, m above the road, s
    # p95 speed (m/s) of the training frames in bins of |curvature| (1/m), the first capped at v_max; read linearly.
    corner_curvature = (0.0, 0.001, 0.0035, 0.0075, 0.015, 0.03, 0.055, 0.095, 0.15)
    corner_speed = (29.0, 22.4, 17.9, 16.4, 14.2, 11.7, 7.8, 6.2, 4.6)
    throttle_keys, brake_keys, hard_brake_keys = {"KeyW"}, {"KeyS"}, {"KeyX"}
    left_keys, right_keys = {"KeyA"}, {"KeyD"}
    hints = ((("KeyW",), "throttle"), (("KeyS",), "brake"), (("KeyX",), "hard brake"), (("KeyA", "KeyD"), "steer"))

    def __init__(self, size: tuple[int, int] = RESOLUTION):
        self.speed = self.curvature = 0.0
        self.size = size

    def curvature_limit(self, speed: float) -> float:
        """The largest |curvature| at this speed: the lateral acceleration budget, within k_max."""
        return min(self.k_max, self.lateral / max(speed, 1.0) ** 2)

    def corner_speed_of(self, curvature: float) -> float:
        """The speed the training data pairs with this |curvature| (its 95th percentile)."""
        return float(np.interp(abs(curvature), self.corner_curvature, self.corner_speed))

    def reset(self, action: np.ndarray) -> None:
        self.speed = float(np.clip(action[0], 0.0, self.v_max))
        limit = self.curvature_limit(self.speed)
        self.curvature = float(np.clip(action[1], -limit, limit))

    def actions(self, keys: set[str], n: int) -> np.ndarray:
        dt, out = 1.0 / FPS, np.zeros((n, 2), dtype=np.float32)
        for j in range(n):
            if keys & self.hard_brake_keys:
                self.speed -= self.hard_brake * dt
            elif keys & self.brake_keys:
                self.speed -= self.brake * dt
            elif keys & self.throttle_keys:  # a power- and drag-limited car: full accel from rest, ~0 at v_max
                self.speed += self.accel * max(0.05, 1.0 - self.speed / self.v_max) * dt
            else:
                self.speed -= self.drag * dt
            self.speed = min(max(self.speed, 0.0), self.v_max)
            target = self.k_max * (bool(keys & self.left_keys) - bool(keys & self.right_keys))
            if target:
                self.curvature += float(np.clip(target - self.curvature, -self.k_rate * dt, self.k_rate * dt))
            else:
                self.curvature *= math.exp(-dt / self.k_return)
            limit = self.curvature_limit(self.speed)
            self.curvature = min(max(self.curvature, -limit), limit)
            corner = self.corner_speed_of(self.curvature)
            if self.speed > corner:
                self.speed = max(corner, self.speed - self.corner_brake * dt)
            out[j] = self.speed, self.curvature
        return out

    def ribbon(self) -> list[tuple[float, float]]:
        """The commanded path: the current speed and curvature held for ribbon_seconds (bicycle model, yaw rate =
        speed x curvature), projected by an approximate pinhole of the front camera (camera_hfov horizontally,
        camera_height above a flat road, the horizon at mid-height). (x, y) pixels of a frame of `size`, from 1 m
        ahead on."""
        h, w = self.size
        focal = (w / 2) / math.tan(math.radians(self.camera_hfov / 2))
        dt, heading, forward, left, points = 1.0 / FPS, 0.0, 0.0, 0.0, []
        for _ in range(round(self.ribbon_seconds * FPS)):
            heading += self.speed * self.curvature * dt
            forward += self.speed * math.cos(heading) * dt
            left += self.speed * math.sin(heading) * dt
            if forward >= 1.0:
                points.append((w / 2 - focal * left / forward, h / 2 + focal * self.camera_height / forward))
        return points

    def hud(self) -> dict:
        ribbon = [(round(x, 1), round(y, 1)) for x, y in self.ribbon()]
        limit = self.curvature_limit(self.speed)
        return {"speed": self.speed, "curvature": self.curvature, "limit": limit, "k_max": self.k_max, "ribbon": ribbon}


class NavigationControls(Controls):
    """RE10K's camera twist [v, w]: v in units of the clip's median speed, w in rad/s, OpenCV axes (x right, y down,
    z forward). Magnitudes stay inside the range of the training clips' twists (normalized per clip, so that the median
    |v| is 1; percentiles below) and cross the thresholds of the keys the world model derives: a direction key fires
    above 0.25, fast above |v| 1.5, slow below 0.5, roll above 0.15 rad/s. Forward at normal speed is |v| 1.4 (p81 of
    |v|), under the fast threshold; with Shift 2.0 (p99)."""

    mouse_look = True
    speeds = {"normal": 1.4, "fast": 2.0, "slow": 0.4}
    moves = {  # key -> (twist dim, value at normal speed)
        "KeyW": (2, 1.0),
        "KeyS": (2, -1.0),
        "KeyD": (0, 0.8),
        "KeyA": (0, -0.8),
        "KeyQ": (1, 0.65),
        "KeyE": (1, -0.65),
    }
    turns = {  # key -> (twist dim, rad/s); +w_y turns right, +w_x looks up (both about p99), +w_z rolls right
        "KeyL": (4, 0.5),
        "KeyJ": (4, -0.5),
        "KeyI": (3, 0.14),
        "KeyK": (3, -0.14),
        "KeyO": (5, 0.2),
        "KeyU": (5, -0.2),
    }
    slow_keys = {"KeyC"}  # not Ctrl: browsers keep Ctrl+W (close the tab) for themselves
    # Mouse look: a pixel of motion turns the camera by this share of what a turn key turns it in a frame (the motion of
    # a step is spread over its frames, so its rate stays constant).
    mouse_share = 0.084
    look_limit = 1.6  # yaw and pitch rates stay within this many turn-key rates: 0.8 and 0.22 rad/s (p99.9)
    hints = (
        (("KeyW", "KeyS"), "forward / back"),
        (("KeyA", "KeyD"), "strafe"),
        (("KeyQ", "KeyE"), "down / up"),
        (("KeyJ", "KeyL"), "yaw"),
        (("KeyI", "KeyK"), "pitch"),
        (("KeyU", "KeyO"), "roll"),
        (("Shift",), "fast"),
        (("KeyC",), "slow"),
    )

    def __init__(self):
        self.pending = np.zeros(2)  # mouse motion (dx, dy) in pixels since the last step

    def reset(self, action: np.ndarray) -> None:
        self.pending[:] = 0.0

    def look(self, dx: float, dy: float) -> None:
        """Mouse motion, accumulated until the next step (dy > 0: down)."""
        self.pending += (dx, dy)

    def actions(self, keys: set[str], n: int) -> np.ndarray:
        pace = "fast" if "Shift" in keys else "slow" if keys & self.slow_keys else "normal"
        twist = np.zeros(6, dtype=np.float32)
        for code in keys & self.moves.keys():
            dim, value = self.moves[code]
            twist[dim] += value * self.speeds[pace]
        for code in keys & self.turns.keys():
            dim, value = self.turns[code]
            twist[dim] += value
        (dx, dy), self.pending[:] = self.pending / n, 0.0
        yaw, pitch = self.turns["KeyL"][1], self.turns["KeyI"][1]
        twist[4] = np.clip(twist[4] + self.mouse_share * yaw * dx, -self.look_limit * yaw, self.look_limit * yaw)
        twist[3] = np.clip(twist[3] - self.mouse_share * pitch * dy, -self.look_limit * pitch, self.look_limit * pitch)
        return np.tile(twist, (n, 1))


class HandControls(Controls):
    """EgoDex's 51 deltas per hand (left first), wrist translation first: a held key moves that wrist at a constant
    speed along an axis of the ARKit world (x right, y up, z back), whose -z is where the head faces within 8 degrees at
    the median (20 at the 90th percentile) of the training episodes. Wrist rotations and finger joints get zero deltas,
    as does a hand without a held key: it holds still."""

    speed = 0.014  # m per frame (28 cm/s), about the 95th-percentile sustained wrist speed (0.5 s means)
    offset = {"left": 0, "right": 51}
    moves = {  # key -> (hand, world axis, sign)
        "KeyA": ("left", 0, -1.0),
        "KeyD": ("left", 0, 1.0),
        "KeyQ": ("left", 1, -1.0),
        "KeyE": ("left", 1, 1.0),
        "KeyW": ("left", 2, -1.0),
        "KeyS": ("left", 2, 1.0),
        "KeyJ": ("right", 0, -1.0),
        "KeyL": ("right", 0, 1.0),
        "KeyU": ("right", 1, -1.0),
        "KeyO": ("right", 1, 1.0),
        "KeyI": ("right", 2, -1.0),
        "KeyK": ("right", 2, 1.0),
    }
    hints = (
        (("KeyQ", "KeyE"), "left down / up"),
        (("KeyW", "KeyS"), "left forward / back"),
        (("KeyA", "KeyD"), "left sideways"),
        (("KeyU", "KeyO"), "right down / up"),
        (("KeyI", "KeyK"), "right forward / back"),
        (("KeyJ", "KeyL"), "right sideways"),
    )

    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def actions(self, keys: set[str], n: int) -> np.ndarray:
        a = np.zeros(self.action_dim, dtype=np.float32)
        for code in keys & self.moves.keys():
            hand, axis, sign = self.moves[code]
            a[self.offset[hand] + axis] += sign * self.speed
        return np.tile(a, (n, 1))


def make_controls(env: str, world_model) -> Controls:
    if env == "robot_manipulation":
        return RobotControls(world_model.config.action_std)
    if env == "driving":
        return DrivingControls()
    if env == "indoor_navigation":
        return NavigationControls()
    return HandControls(world_model.config.action_dim)


class Rollout:
    """One environment rolling out from a prompt clip (an EnvironmentDataset item of CLIP_FRAMES frames): the world
    model's KV cache of the last n_ctx latents, the streaming decoder, and the position in the clip's recorded
    actions, which advances tau frames a step whatever drives it."""

    def __init__(self, world_model, autoencoder, clip: dict, index: int, seed: int, device: torch.device):
        self.world_model, self.autoencoder, self.tau = world_model, autoencoder, world_model.config.tau
        self.key, self.index, self.seed, self.device = clip["key"], index, seed, device
        self.generator = torch.Generator(device).manual_seed(seed)
        self.recorded = clip["actions"].numpy()
        n_ctx = CONTEXT_FRAMES // self.tau
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = encode_video(autoencoder, clip["video"][None, :CONTEXT_FRAMES].to(device))
            actions = torch.from_numpy(self.recorded[None]).to(device)  # encode_actions slices the context's
            self.kv = world_model.context_cache(z, world_model.encode_actions(actions, 0, n_ctx + 1)[:, :-1])
            self.decoder = StreamingDecoder(
                autoencoder, z, ROLLOUT["decode_steps"], ROLLOUT["decode_alpha"], self.generator
            )
        self.position = CONTEXT_FRAMES - 1  # the first action of the first generated latent
        self.last_action = self.recorded[self.position - 1]
        self.steps = 0

    def randn(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device, dtype=dtype)

    def recorded_actions(self) -> np.ndarray | None:
        """The clip's actions of the next step, None past their end (replay never wraps)."""
        if self.position + self.tau > len(self.recorded):
            return None
        return self.recorded[self.position : self.position + self.tau]

    def remaining(self) -> int:
        return (len(self.recorded) - self.position) // self.tau

    @torch.no_grad()
    def step(self, actions: np.ndarray, denoise_steps: int, context_noise: float) -> torch.Tensor:
        """Generate the next latent under actions (tau, action_dim) and decode its frames: uint8 (tau, 3, H, W)."""
        wm = self.world_model
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            # A latent's action embedding depends on its own tau actions only (after the initial token).
            a = wm.action_encoder(torch.from_numpy(actions[None]).to(self.device))[:, 1:]
            shape, dtype = (1, 1, *self.decoder.window.shape[2:]), self.decoder.window.dtype
            x, self.kv = wm.next_latent(
                self.kv, a, self.randn(shape, dtype), self.randn(shape, dtype), denoise_steps, context_noise
            )
            video = self.decoder.push(x)[0]
        self.position += self.tau
        self.last_action = actions[-1]
        self.steps += 1
        return (video.float().clamp(0, 1) * 255).round().to(torch.uint8)


def to_jpeg(frames: torch.Tensor) -> list[bytes]:
    """uint8 (T, 3, H, W) -> JPEG files (encoded on the frames' device)."""
    return [bytes(jpeg.cpu().numpy()) for jpeg in encode_jpeg(list(frames), quality=JPEG_QUALITY)]


@dataclass
class Chunk:
    """Frames to play, with what the page shows about them."""

    frames: list[bytes]
    info: dict
    started: float  # perf_counter when the step read its actions


def pack(header: dict, jpeg: bytes) -> bytes:
    """A binary frame message: the header's length (uint32 LE), the header as JSON, the JPEG."""
    text = json.dumps(header).encode()
    return struct.pack("<I", len(text)) + text + jpeg


class Demo:
    """The world models of one autoencoder, their prompt datasets and the state of the one browser session. The GPU
    work runs in one worker thread (produce), which also applies the page's commands, in their order."""

    def __init__(
        self,
        autoencoder: CADRE,
        name: str,
        world_models: dict,
        datasets: dict[str, EnvironmentDataset],
        device: torch.device,
        denoise_steps: int,
        context_noise: float,
    ):
        self.autoencoder, self.name, self.world_models, self.datasets = autoencoder, name, world_models, datasets
        self.device, self.denoise_steps, self.context_noise = device, denoise_steps, context_noise
        self.envs = list(world_models)
        self.env = self.envs[0]
        self.controls = {env: make_controls(env, wm) for env, wm in world_models.items()}
        self.modes = {env: MODES[0] for env in self.envs}
        self.rollouts: dict[str, Rollout] = {}
        self.held: set[str] = set()
        self.tapped: set[str] = set()  # keys pressed since the last step, so that a short tap still acts
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = asyncio.Lock()
        self.socket = None
        self.rng = random.Random()  # the rollouts' noise seeds, from system entropy; reseeded per session

    # --- worker thread ---

    def new_prompt(self, env: str, index: int, seed: int | None = None) -> None:
        """Start env's rollout from a window of its dataset."""
        dataset = self.datasets[env]
        index %= len(dataset)
        seed = self.rng.randrange(2**31) if seed is None else seed
        rollout = Rollout(self.world_models[env], self.autoencoder, dataset[index], index, seed, self.device)
        self.rollouts[env] = rollout
        self.controls[env].reset(rollout.last_action)

    def apply(self, command: dict) -> None:
        """Apply a command of the page."""
        kind, env = command["type"], self.env
        if kind == "env" and command["env"] in self.world_models:  # the page releases its keys first
            self.env = command["env"]
        elif kind == "prompt":  # a window of the dataset, in the current mode
            self.new_prompt(env, int(command["index"]), command.get("seed"))
        elif kind == "next":  # the next window, in the current mode
            self.new_prompt(env, self.next_prompt(env))
        elif kind == "replay":  # the prompt clip again, under its recorded actions
            self.modes[env] = "replay"
            index = self.rollouts[env].index if env in self.rollouts else self.next_prompt(env)
            self.new_prompt(env, command.get("index", index), command.get("seed"))
        elif kind == "mode" and command["mode"] in MODES:
            self.modes[env] = command["mode"]
            if command["mode"] == "interactive" and env in self.rollouts:
                self.controls[env].reset(self.rollouts[env].last_action)
        elif kind == "press":
            self.controls[env].press(command["code"])
        elif kind == "look" and self.controls[env].mouse_look and self.modes[env] == "interactive":
            self.controls[env].look(float(command["dx"]), float(command["dy"]))
        elif kind == "joint" and isinstance(self.controls[env], RobotControls):
            self.controls[env].select(command["arm"], command["joint"])
        elif kind == "settings":
            self.denoise_steps = min(max(int(command.get("denoise_steps", self.denoise_steps)), 1), 50)
            self.context_noise = min(max(float(command.get("context_noise", self.context_noise)), 0.0), 1.0)
        return None

    def next_prompt(self, env: str) -> int:
        """The window of env's dataset after the current one, cycling; the first one before any."""
        return (self.rollouts[env].index + 1) % len(self.datasets[env]) if env in self.rollouts else 0

    def produce(self) -> Chunk:
        """Apply the pending commands, then the frames of one step of the current environment (from its first window
        if it has no rollout). A replay at the end of its recording goes on with the next window's."""
        while not self.commands.empty():
            self.apply(self.commands.get())
        env = self.env
        if env not in self.rollouts:
            self.new_prompt(env, self.next_prompt(env))
        if self.modes[env] == "replay" and self.rollouts[env].recorded_actions() is None:
            self.new_prompt(env, self.next_prompt(env))
        rollout, started = self.rollouts[env], time.perf_counter()
        if self.modes[env] == "replay":
            actions = rollout.recorded_actions()
        else:
            keys, self.tapped = self.held | self.tapped, set()
            actions = self.controls[env].actions(keys, rollout.tau)
        frames = to_jpeg(rollout.step(actions, self.denoise_steps, self.context_noise))
        compute_ms = 1000 * (time.perf_counter() - started)
        return Chunk(frames, self.info(env, compute_ms), started)

    def info(self, env: str, compute_ms: float | None = None) -> dict:
        rollout = self.rollouts[env]
        return {
            "env": env,
            "mode": self.modes[env],
            "index": rollout.index,
            "step": rollout.steps,
            "remaining": rollout.remaining(),
            "prompt": rollout.key,
            "seed": rollout.seed,
            "compute_ms": compute_ms,
            "denoise_steps": self.denoise_steps,
            "context_noise": self.context_noise,
            "hud": self.controls[env].hud(),
        }

    # --- event loop ---

    def hello(self) -> dict:
        envs = []
        for env in self.envs:
            controls = self.controls[env]
            hints = [[list(codes), what] for codes, what in controls.hints]
            joints = list(RobotControls.joints) if isinstance(controls, RobotControls) else None
            envs.append(
                {
                    "name": env,
                    "title": TITLES[env],
                    "modes": MODES,
                    "hints": hints,
                    "look": controls.mouse_look,
                    "joints": joints,
                }
            )
        return {
            "type": "hello",
            "tau": self.autoencoder.config.tau,
            "fps": FPS,
            "env": self.env,
            "envs": envs,
            "denoise_steps": self.denoise_steps,
            "context_noise": self.context_noise,
        }

    async def session(self, socket) -> None:
        """Stream to an accepted websocket until it closes or a newer session takes over."""
        self.socket = socket
        async with self.lock:
            if self.socket is not socket:
                return
            self.rng = random.Random()
            self.env, self.rollouts = self.envs[0], {}  # a new viewer starts on each environment's first clip
            await socket.send_text(json.dumps(self.hello()))
            receiver = asyncio.create_task(self.receive(socket))
            try:
                await self.stream(socket, receiver)
            finally:
                receiver.cancel()
            if self.socket is not socket:
                await socket.close(code=4000, reason="another viewer took over")
            elif receiver.done() and not receiver.cancelled():
                receiver.result()  # the disconnect, or an error of the receiver

    async def receive(self, socket) -> None:
        while True:
            message = json.loads(await socket.receive_text())
            if message["type"] == "keys":
                self.held = set(message["held"])
                if message.get("pressed"):
                    self.tapped.add(message["pressed"])
                    self.commands.put({"type": "press", "code": message["pressed"]})
            else:
                self.commands.put(message)

    async def stream(self, socket, receiver: asyncio.Task) -> None:
        """Play each chunk at FPS while the worker produces the next."""
        loop = asyncio.get_running_loop()
        clock, playing = loop.time(), None
        while not receiver.done() and self.socket is socket:
            work = loop.run_in_executor(self.executor, self.produce)
            if playing is not None:
                clock = await self.play(socket, playing, clock)
            playing = await work

    async def play(self, socket, chunk: Chunk, clock: float) -> float:
        """Send the chunk's frames one per 1 / FPS s from `clock` (or now, if production fell behind)."""
        loop = asyncio.get_running_loop()
        clock = max(clock, loop.time())
        for i, jpeg in enumerate(chunk.frames):
            await asyncio.sleep(max(0.0, clock - loop.time()))
            header = {"i": i, "n": len(chunk.frames)}
            if i == 0:
                header |= chunk.info | {"latency_ms": 1000 * (time.perf_counter() - chunk.started)}
            await socket.send_bytes(pack(header, jpeg))
            clock += 1.0 / FPS
        return clock


def same_autoencoder(a: dict, b: dict) -> bool:
    """Whether two release autoencoder checkpoints hold identical weights, statistics and config."""
    if a["config"] != b["config"] or a["state_dict"].keys() != b["state_dict"].keys():
        return False
    stats = all(torch.equal(a[k], b[k]) for k in ("latent_mean", "latent_std"))
    return stats and all(torch.equal(v, b["state_dict"][k]) for k, v in a["state_dict"].items())


def load_models(name: str, checkpoints: dict, device: torch.device) -> tuple[CADRE, dict]:
    """The world models of `checkpoints` (environment -> release world-model checkpoint, a path or a file object)
    and the one autoencoder they nest, built once: every world model's copy must equal the first's."""
    autoencoder, first, world_models = None, None, {}
    for env, checkpoint in checkpoints.items():
        world_model, nested = load_checkpoint(checkpoint, device)
        assert nested["model"] == name, f"{env}: a world model of {nested['model']}, not {name}"
        assert world_model.config.action_dim == ENVIRONMENTS[env].action_dim, f"{env}: not a world model of {env}"
        if first is None:
            with torch.device(device):
                autoencoder = CADRE.from_checkpoint(nested)
            first = nested
        assert same_autoencoder(nested, first), f"{env}: its autoencoder differs from {next(iter(checkpoints))}'s"
        assert world_model.config.tau == autoencoder.config.tau
        world_models[env] = world_model.eval()
        print(f"loaded {env} world model ({name}, autoencoder identical to the shared one)", flush=True)
    return autoencoder, world_models


def warm_up(demo: Demo, steps: int = 4) -> dict[str, float]:
    """Load a prompt and run a few steps in every environment (compiles, if enabled); returns the ms of the last
    step of each and leaves no rollout behind."""
    timings = {}
    for env in demo.envs:
        demo.env = env
        demo.new_prompt(env, index=0, seed=0)
        for _ in range(steps):
            timings[env] = demo.produce().info["compute_ms"]
    demo.env, demo.rollouts = demo.envs[0], {}
    return timings


def parse_pairs(pairs: list[str]) -> dict[str, str]:
    """["env=value", ...] -> {env: value}."""
    out = dict(p.split("=", 1) for p in pairs)
    assert set(out) <= set(ENVIRONMENTS), f"unknown environments {set(out) - set(ENVIRONMENTS)}"
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--autoencoder", choices=AUTOENCODERS, default=AUTOENCODERS[0])
    parser.add_argument(
        "--weights",
        default="kyutai/cadre-world-models-2x16x16",
        help="Hub repo id or folder of <environment>/ world models, or a directory of <environment>/<autoencoder>.pt",
    )
    parser.add_argument(
        "--data", nargs="+", metavar="ENV=ROOT", help="prompt clips (DATA.md); default: the demo_clips/ of --weights"
    )
    parser.add_argument("--manifest", nargs="*", default=[], metavar="ENV=PATH", help="their windows")
    parser.add_argument("--split", default="test", help="the dataset split of the prompt clips")
    parser.add_argument(
        "--envs",
        nargs="+",
        choices=list(ENVIRONMENTS),
        default=["robot_manipulation", "human_manipulation", "driving", "indoor_navigation"],
    )
    parser.add_argument("--denoise_steps", type=int, help=f"default: {DENOISE_STEPS}")
    parser.add_argument("--context_noise", type=float, default=CONTEXT_NOISE)
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile, with CUDA graphs"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args(argv)


def build(args: argparse.Namespace, checkpoints: dict | None = None) -> Demo:
    """The demo of the parsed arguments; `checkpoints` (environment -> path or file object) replaces --weights."""
    device = torch.device(args.device)
    if checkpoints is None:
        pt = {env: Path(args.weights) / env / f"{args.autoencoder}.pt" for env in args.envs}
        checkpoints = {env: path if path.is_file() else f"{args.weights}/{env}" for env, path in pt.items()}
    autoencoder, world_models = load_models(args.autoencoder, checkpoints, device)
    if args.data:
        data, manifests = parse_pairs(args.data), parse_pairs(args.manifest)
    else:  # six clips per environment, shipped with the world models
        clips = Path(args.weights) / "demo_clips"
        if not clips.is_dir():
            clips = Path(snapshot_download(args.weights, allow_patterns=["demo_clips/*"])) / "demo_clips"
        data = {env: clips / env for env in world_models}
        manifests = {env: clips / env / "manifest.txt" for env in world_models}
    resolution = autoencoder.config.resolution
    datasets = {
        env: EnvironmentDataset(env, data[env], args.split, CLIP_FRAMES, resolution, manifests.get(env))
        for env in world_models
    }
    if args.compile:  # CUDA graphs of static shapes: a context, a step and a cache update per world model, a window
        torch._dynamo.config.recompile_limit = 8 * len(world_models)
        compile = dict(mode="reduce-overhead", dynamic=False)
        for world_model in world_models.values():
            world_model.velocity = torch.compile(world_model.velocity, **compile)
        autoencoder.decoder.flow_forward = torch.compile(autoencoder.decoder.flow_forward, **compile)
    denoise_steps = args.denoise_steps or DENOISE_STEPS[args.autoencoder]
    return Demo(autoencoder, args.autoencoder, world_models, datasets, device, denoise_steps, args.context_noise)


def serve(demo: Demo, host: str, port: int) -> None:
    """Serve the page and its websocket (the demo extra: starlette, uvicorn, websockets)."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    page = (HERE / "demo.html").read_text()

    async def index(request):
        return HTMLResponse(page)

    async def websocket(socket):
        await socket.accept()
        try:
            await demo.session(socket)
        except WebSocketDisconnect:
            pass

    app = Starlette(routes=[Route("/", index), WebSocketRoute("/ws", websocket)])
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    demo = build(args)
    # In the worker thread, which runs all the GPU work: CUDA graphs are recorded per thread.
    for env, ms in demo.executor.submit(warm_up, demo).result().items():
        tau = demo.autoencoder.config.tau
        print(f"{env}: {ms:.0f} ms per step of {tau} frames = {1000 * tau / ms:.1f} fps (real time: {FPS})", flush=True)
    print(f"serving {args.autoencoder} ({', '.join(demo.envs)}) on http://{args.host}:{args.port}", flush=True)
    serve(demo, args.host, args.port)


if __name__ == "__main__":
    main()
