"""Interactive demo of the CADRE world models: drive them from a browser, at the video's 20 fps.

    uv sync --extra demo
    uv run python demo.py

then open http://localhost:8000 (on a cluster node: ssh -N -L 8000:<node>:8000 <login host>).

One process hosts the world models of one autoencoder, <weights>/<environment> for each environment of --envs (by
default the Hub repo kyutai/cadre-world-models-2x16x16; a local directory of <environment>/<autoencoder>.pt files also
works), sharing one autoencoder and serving --max_active independent browser sessions over websockets:
the others wait in line (first come, first served), told their place and wait. A turn lasts as long as no one waits;
once someone does, it ends after --turn_seconds from its start, or after --idle_seconds without input. Each
environment rolls out from a real prompt: the first 40 frames of a window of its --data root (DATA.md; the --manifest
windows, else the split's; by default the six clips per environment of the weights' demo_clips/), which also brings the
recorded actions of the replay mode. The prompt only primes the world model (its latents and their KV cache): the page
shows the generated frames alone, from the first step on, as the paper's demo did.
A step reads the current action, predicts the next latent with the world model's KV cache (next_latent: denoise_steps
Euler steps, the new frame cached at context noise), decodes its tau frames as the rollout evaluation streams them (a
window of latents sliding one latent a step, the decoder's noise an AR(1) buffer), and sends them as JPEG, paced at 20
fps; the next step runs while they play. All of it under bf16 autocast, as the rollout evaluation runs, with the linear
layers' weights stored in bf16 (autocast would cast them to it on every call: 30% of a step) and the others in fp32,
compiled with CUDA graphs (a few minutes at startup; --no-compile runs eager, about 2.5x slower). The default
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
import contextlib
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
import queue
import random
import secrets
import struct
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from torchvision.io import encode_jpeg

from dataset import ENVIRONMENTS, FPS, RESOLUTION, EnvironmentDataset
from demo_batch import (
    batch_key,
    decode_batches,
    decoder_batch_key,
    prepare_batch,
    step_batch,
)
from demo_scheduler import BatchScheduler
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
CONTEXT_NOISE_BY_ENV = {
    "driving": 0.35,
    "indoor_navigation": 0.35,
    "robot_manipulation": 0.20,
    "human_manipulation": 0.20,
}

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

    def __init__(
        self,
        world_model,
        autoencoder,
        clip: dict,
        index: int,
        seed: int,
        device: torch.device,
        prepared=None,
        decoder_context_frames: int | None = None,
    ):
        self.world_model, self.autoencoder, self.tau = world_model, autoencoder, world_model.config.tau
        self.key, self.index, self.seed, self.device = clip["key"], index, seed, device
        self.generator = torch.Generator(device).manual_seed(seed)
        self.recorded = clip["actions"].numpy()
        n_ctx = CONTEXT_FRAMES // self.tau
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if prepared is None:
                z = encode_video(autoencoder, clip["video"][None, :CONTEXT_FRAMES].to(device))
                actions = torch.from_numpy(self.recorded[None]).to(device)
                self.kv = world_model.context_cache(z, world_model.encode_actions(actions, 0, n_ctx + 1)[:, :-1])
            else:
                cached_z, cached_kv = prepared
                z = cached_z.to(device, copy=True)
                self.kv = [None if pair is None else tuple(t.to(device, copy=True) for t in pair) for pair in cached_kv]
            self.decoder = StreamingDecoder(
                autoencoder,
                z,
                ROLLOUT["decode_steps"],
                ROLLOUT["decode_alpha"],
                self.generator,
                context_frames=decoder_context_frames,
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
    preview: bytes | None = None  # for the line, when someone waits


def pack(header: dict, jpeg: bytes) -> bytes:
    """A binary frame message: the header's length (uint32 LE), the header as JSON, the JPEG."""
    text = json.dumps(header).encode()
    return struct.pack("<I", len(text)) + text + jpeg


class Demo:
    """The world models of one autoencoder, their prompt datasets, the state of the browser session whose turn it is
    and the line of the others. The GPU work runs in one worker thread (produce), which also applies the page's
    commands, in their order."""

    def __init__(
        self,
        autoencoder: CADRE,
        name: str,
        world_models: dict,
        datasets: dict[str, EnvironmentDataset],
        device: torch.device,
        denoise_steps: int,
        context_noise: float | None = None,
        turn_seconds: float = 120.0,
        idle_seconds: float = 30.0,
        max_line: int = 100,
        log: Path | None = None,
        prompt_cache: int = 6,
        fixed_settings: bool = False,
        decoder_context_frames: int | None = None,
    ):
        self.autoencoder, self.name, self.world_models, self.datasets = autoencoder, name, world_models, datasets
        self.device, self.defaults = device, (denoise_steps, context_noise)
        self.context_noise_defaults = {
            env: CONTEXT_NOISE_BY_ENV[env] if context_noise is None else context_noise for env in world_models
        }
        self.fixed_settings = fixed_settings
        self.decoder_context_frames = StreamingDecoder.validate_context_frames(
            autoencoder.config, decoder_context_frames
        )
        self.turn_seconds, self.idle_seconds, self.max_line = turn_seconds, idle_seconds, max_line
        self.log, self.salt = log, None
        if log is not None:  # the visitors' ids hash their address with a salt kept beside the log, never stored
            salt = log.with_name(log.name + ".salt")
            if not salt.exists():
                salt.write_text(secrets.token_hex(16))
            self.salt = salt.read_text().strip()
        self.prompt_cache_size = prompt_cache
        self.prompt_cache = OrderedDict()  # CPU tensors shared by slots; only accessed by the GPU worker
        self.envs = list(world_models)
        self.commands: queue.SimpleQueue = queue.SimpleQueue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.scheduler = None
        self.line: list = []  # the sockets of the sessions, the one whose turn it is (or is next) first
        self.turn_started = self.last_input = 0.0  # event loop times
        self.preview, self.fresh = None, asyncio.Event()  # the stream's latest preview, and the event of the next
        self.reset()

    def reset(self) -> None:
        """A new session's state: each environment's first clip, new controls, the default mode and settings, no key
        held."""
        self.env, self.rollouts = self.envs[0], {}
        self.controls = {env: make_controls(env, wm) for env, wm in self.world_models.items()}
        self.played = dict.fromkeys(self.envs, 0)  # the steps of the session in each environment
        self.modes = {env: MODES[0] for env in self.envs}
        self.denoise_steps = self.defaults[0]
        self.context_noises = self.context_noise_defaults.copy()
        self.input_sequence = None
        self.preview_encoded_at: float | None = None
        self.held: set[str] = set()
        self.tapped: set[str] = set()  # keys pressed since the last step, so that a short tap still acts
        while not self.commands.empty():
            self.commands.get()
        self.rng = random.Random()  # the rollouts' noise seeds, from system entropy

    @property
    def context_noise(self) -> float:
        """Current environment's setting, including direct environment selection during warmup."""
        return self.context_noises[self.env]

    @context_noise.setter
    def context_noise(self, value: float) -> None:
        self.context_noises[self.env] = value

    # --- worker thread ---

    def new_prompt(self, env: str, index: int, seed: int | None = None) -> None:
        """Start env's rollout from a window of its dataset."""
        dataset = self.datasets[env]
        index %= len(dataset)
        seed = self.rng.randrange(2**31) if seed is None else seed
        key = (env, index)
        cached = self.prompt_cache.get(key)
        clip = dataset[index] if cached is None else cached[0]
        prepared = None if cached is None else cached[1]
        rollout = Rollout(
            self.world_models[env],
            self.autoencoder,
            clip,
            index,
            seed,
            self.device,
            prepared,
            decoder_context_frames=self.decoder_context_frames,
        )
        if self.prompt_cache_size:
            if cached is None:
                # Retain deterministic context only. Each session draws its own decoder noise.
                z = rollout.decoder.window.detach().to("cpu", copy=True)
                kv = [
                    None if pair is None else tuple(t.detach().to("cpu", copy=True) for t in pair)
                    for pair in rollout.kv
                ]
                metadata = {"key": clip["key"], "actions": clip["actions"]}
                self.prompt_cache[key] = (metadata, (z, kv))
            self.prompt_cache.move_to_end(key)
            while len(self.prompt_cache) > self.prompt_cache_size:
                self.prompt_cache.popitem(last=False)
        self.rollouts[env] = rollout
        self.controls[env].reset(rollout.last_action)

    def apply(self, command: dict) -> None:
        """Apply a command of the page."""
        kind, env = command["type"], self.env
        if kind == "keys":
            self.input_sequence = command.get("sequence")
            self.held = set(command["held"])
            if command.get("pressed"):
                self.tapped.add(command["pressed"])
                self.controls[env].press(command["pressed"])
        elif kind == "env" and command["env"] in self.world_models:  # the page releases its keys first
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
        elif kind == "settings" and not self.fixed_settings:
            self.denoise_steps = min(max(int(command.get("denoise_steps", self.denoise_steps)), 1), self.defaults[0])
            self.context_noise = min(max(float(command.get("context_noise", self.context_noise)), 0.0), 1.0)
        return None

    def next_prompt(self, env: str) -> int:
        """The window of env's dataset after the current one, cycling; the first one before any."""
        return (self.rollouts[env].index + 1) % len(self.datasets[env]) if env in self.rollouts else 0

    def prepare_step(self) -> tuple:
        """Apply the pending commands, then the frames of one step of the current environment (from its first window
        if it has no rollout). A replay at the end of its recording goes on with the next window's."""
        worker_started = time.perf_counter()
        while not self.commands.empty():
            self.apply(self.commands.get())
        env = self.env
        if env not in self.rollouts:
            self.new_prompt(env, self.next_prompt(env))
        if self.modes[env] == "replay" and self.rollouts[env].recorded_actions() is None:
            self.new_prompt(env, self.next_prompt(env))
        rollout, started = self.rollouts[env], time.perf_counter()
        input_sequence = self.input_sequence
        self.played[env] += 1
        if self.modes[env] == "replay":
            actions = rollout.recorded_actions()
        else:
            keys, self.tapped = self.held | self.tapped, set()
            actions = self.controls[env].actions(keys, rollout.tau)
        return env, actions, worker_started, started, input_sequence

    def finish_step(
        self, prepared: tuple, video: torch.Tensor, batch_size: int = 1, *, frames: list[bytes] | None = None
    ) -> Chunk:
        env, _, worker_started, started, input_sequence = prepared
        if frames is None:
            frames = to_jpeg(video)
        preview = None
        if len(self.line) > 1:
            now = time.perf_counter()
            if self.preview_encoded_at is None or now - self.preview_encoded_at >= 0.25:
                # Reuse the active JPEG: waiting viewers scale it in the browser,
                # so their preview adds no GPU resize or encoding work.
                preview = frames[0]
                self.preview_encoded_at = now
        compute_ms = 1000 * (time.perf_counter() - started)
        info = self.info(env, compute_ms) | {
            "input_sequence": input_sequence,
            "worker_started": worker_started,
            "prompt_ms": 1000 * (started - worker_started),
            "batch_size": batch_size,
        }
        return Chunk(frames, info, started, preview)

    def produce(self) -> Chunk:
        prepared = self.prepare_step()
        env, actions = prepared[:2]
        video = self.rollouts[env].step(actions, self.denoise_steps, self.context_noise)
        return self.finish_step(prepared, video)

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
            "decoder_context_frames": self.decoder_context_frames,
            "cache_decoder_prefix": getattr(self, "cache_decoder_prefix", False),
            "short_decoder_attention": getattr(self, "short_decoder_attention", False),
            "tiny_decoder_attention": getattr(self, "tiny_decoder_attention", False),
            "fused_swiglu": getattr(self, "fused_swiglu", False),
            "fused_swiglu_world_model_only": getattr(self, "fused_swiglu_world_model_only", False),
            "preproject_conditioning": getattr(self, "preproject_conditioning", False),
            "precision": getattr(self, "precision", "bf16"),
            "world_model_precision": getattr(self, "world_model_precision", "bf16"),
            "decoder_precision": getattr(self, "decoder_precision", "bf16"),
            "fp8_decoder_only": getattr(self, "fp8_decoder_only", False),
            "context_noise": self.context_noises[env],
            "context_noise_by_env": self.context_noises.copy(),
            "hud": self.controls[env].hud(),
        }

    # --- event loop ---

    def concurrency(self) -> dict:
        """Global admission counts shared by active streams, waiters and health."""
        pool = getattr(self, "pool", None)
        slots = pool.slots if pool is not None else [self]
        by_env = dict.fromkeys(self.envs, 0)
        for slot in slots:
            if slot.line:
                by_env[slot.env] += 1
        return {
            "active": sum(by_env.values()),
            "max_active": len(slots),
            "waiting": sum(max(0, len(slot.line) - 1) for slot in slots),
            "by_env": by_env,
        }

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
                    "context_noise": self.context_noises[env],
                }
            )
        return {
            "type": "hello",
            "tau": self.autoencoder.config.tau,
            "fps": FPS,
            "env": self.env,
            "envs": envs,
            "denoise_steps": self.denoise_steps,
            "decoder_context_frames": self.decoder_context_frames,
            "cache_decoder_prefix": getattr(self, "cache_decoder_prefix", False),
            "short_decoder_attention": getattr(self, "short_decoder_attention", False),
            "tiny_decoder_attention": getattr(self, "tiny_decoder_attention", False),
            "fused_swiglu": getattr(self, "fused_swiglu", False),
            "fused_swiglu_world_model_only": getattr(self, "fused_swiglu_world_model_only", False),
            "preproject_conditioning": getattr(self, "preproject_conditioning", False),
            "context_noise": self.context_noise,
            "context_noise_by_env": self.context_noises.copy(),
            "max_active": getattr(self, "max_active", 1),
            "concurrency": self.concurrency(),
            "max_denoise_steps": self.defaults[0],
            "settings_locked": self.fixed_settings,
            "precision": getattr(self, "precision", "bf16"),
            "world_model_precision": getattr(self, "world_model_precision", "bf16"),
            "decoder_precision": getattr(self, "decoder_precision", "bf16"),
            "fp8_decoder_only": getattr(self, "fp8_decoder_only", False),
        }

    async def session(self, socket, *, reserved: bool = False, joined: float | None = None) -> None:
        """Wait in line on an accepted websocket, then stream to it until it closes or its turn ends."""
        loop = asyncio.get_running_loop()
        now = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        visit = {"time": now, "visitor": self.visitor(socket), "place": len(self.line)}
        joined = loop.time() if joined is None else joined
        if not reserved:
            if len(self.line) > self.max_line:
                self.record(visit | {"outcome": "line_full"})
                await socket.close(code=4002, reason="the line is full")
                return
            self.line.append(socket)
        visit["outcome"] = "left_in_line"
        migrated = False
        try:
            turn = await self.wait_turn(socket)
            if isinstance(turn, Demo):
                migrated = True
                await turn.session(socket, reserved=True, joined=joined)
                return
            if not turn:
                return
            self.turn_started = self.last_input = loop.time()
            visit |= {"waited_s": round(self.turn_started - joined, 1), "outcome": "left"}
            await loop.run_in_executor(self.executor, self.reset)  # after the last session's step, if in flight
            await socket.send_text(json.dumps(self.hello()))
            receiver = asyncio.create_task(self.receive(socket))
            try:
                await self.stream(socket, receiver)
            finally:
                receiver.cancel()
            if receiver.done() and not receiver.cancelled():
                receiver.result()  # the disconnect, or an error of the receiver
            visit["outcome"] = "turn_over"
            await socket.send_text(json.dumps({"type": "over"}))
            await socket.close(code=4001, reason="your turn is over")
        finally:
            if not migrated:  # the destination owns cleanup and the single visit log after a transfer
                self.line.remove(socket)
                if "waited_s" in visit:
                    visit["turn_s"] = round(loop.time() - self.turn_started, 1)
                    tau = self.autoencoder.config.tau
                    visit["seconds"] = {env: n * tau / FPS for env, n in self.played.items() if n}  # of video, per env
                else:
                    visit["waited_s"] = round(loop.time() - joined, 1)
                self.record(visit)

    def visitor(self, socket) -> str | None:
        """An id of the socket's address (behind the proxy, the forwarded one): a salted hash, not the address."""
        if self.salt is None or socket.client is None:
            return None
        return hashlib.sha256(f"{self.salt}:{socket.client.host}".encode()).hexdigest()[:16]

    def record(self, visit: dict) -> None:
        """Append a visit to the log (one JSON line), if there is one."""
        if self.log is not None:
            with self.log.open("a") as f:
                f.write(json.dumps(visit) + "\n")

    def wait(self, place: int) -> float:
        """The longest wait (s) of the place-th session in line (1: next), were every turn to go its full length."""
        elapsed = asyncio.get_running_loop().time() - self.turn_started
        return max(0.0, self.turn_seconds - elapsed) + (place - 1) * self.turn_seconds

    def turn_over(self) -> bool:
        """Whether the current turn ends: someone waits, and it has lasted its length or had no input for a while."""
        if len(self.line) < 2:
            return False
        now = asyncio.get_running_loop().time()
        return now - self.turn_started >= self.turn_seconds or now - self.last_input >= self.idle_seconds

    async def wait_turn(self, socket) -> "bool | Demo":
        """Wait for this slot (True), a reserved idle slot (Demo), or a disconnect (False), sending queue status and
        the stream's latest preview while waiting."""

        async def closed():
            while (await socket.receive())["type"] != "websocket.disconnect":
                pass  # the page sends nothing that matters before its turn

        watch, last, shown = asyncio.create_task(closed()), None, None
        try:
            while self.line[0] is not socket:
                if watch.done():
                    return False
                pool = getattr(self, "pool", None)
                if pool is not None and pool.idle_slot(self, socket) is not None:
                    # Finish the old receive before handing the socket to another
                    # slot, then recheck availability after the await.
                    watch.cancel()
                    await asyncio.gather(watch, return_exceptions=True)
                    if not watch.cancelled():
                        return False
                    destination = pool.move_waiter(self, socket)
                    if destination is not None:
                        return destination
                    watch = asyncio.create_task(closed())
                    continue
                place = self.line.index(socket)
                status = {
                    "type": "line",
                    "place": place,
                    "length": len(self.line) - 1,
                    "wait": round(self.wait(place)),
                    "concurrency": self.concurrency(),
                }
                if status != last:
                    await socket.send_text(json.dumps(status))
                    last = status
                if self.preview is not None and self.preview is not shown:
                    shown = self.preview
                    await socket.send_bytes(pack({"preview": True}, shown))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.fresh.wait(), 0.25)
            return not watch.done()
        finally:
            watch.cancel()
            if not watch.done():
                await asyncio.gather(watch, return_exceptions=True)

    async def receive(self, socket) -> None:
        while True:
            message = json.loads(await socket.receive_text())
            self.last_input = asyncio.get_running_loop().time()
            self.commands.put(message)

    async def stream(self, socket, receiver: asyncio.Task) -> None:
        """Play at FPS and submit the next step just early enough to meet its first-frame deadline."""
        loop = asyncio.get_running_loop()
        clock, work, playback = loop.time(), None, None
        recent_service = []

        def submit():
            submitted = time.perf_counter()
            work = (
                self.scheduler.submit(self)
                if self.scheduler is not None
                else loop.run_in_executor(self.executor, self.produce)
            )
            return submitted, work

        try:
            if self.scheduler is not None:
                self.scheduler.activate(self)
            submitted, work = submit()
            while not receiver.done() and not self.turn_over():
                playing = await asyncio.shield(work)
                worker_started = playing.info.pop("worker_started", playing.started)
                playing.info["queue_ms"] = 1000 * (worker_started - submitted)
                playing.started = submitted
                compute_ms = playing.info.get("compute_ms")
                service = (
                    time.perf_counter() - submitted
                    if compute_ms is None
                    else (compute_ms + playing.info["queue_ms"] + playing.info.get("prompt_ms", 0.0)) / 1000
                )
                recent_service = (recent_service + [max(0.0, service)])[-3:]
                if receiver.done() or self.turn_over():
                    break

                clock = max(clock, loop.time())
                deadline = clock + len(playing.frames) / FPS
                playback = asyncio.create_task(self.play(socket, playing, clock))
                # Leave input collection open until the last useful moment. A
                # short recent maximum absorbs service jitter without building
                # a queue of video whose actions were sampled too early.
                delay = max(0.0, deadline - max(recent_service) - 0.002 - loop.time())
                if self.scheduler is not None:
                    self.scheduler.plan(self, loop.time() + delay)
                if delay:
                    await asyncio.wait((receiver, playback), timeout=delay, return_when=asyncio.FIRST_EXCEPTION)
                if receiver.done() or self.turn_over():
                    break
                if playback.done():
                    playback.result()  # propagate a send failure before scheduling more inference
                submitted, work = submit()
                await asyncio.wait((receiver, playback), return_when=asyncio.FIRST_COMPLETED)
                if receiver.done():
                    break
                clock = playback.result()
                playback = None
        finally:
            if self.scheduler is not None:
                self.scheduler.deactivate(self)
            if playback is not None:
                playback.cancel()
                await asyncio.gather(playback, return_exceptions=True)
            # A departed user's pending step must finish before this slot is reset/reused.
            if work is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(work)

    def turn(self) -> dict:
        """What the page shows of its turn: the sessions waiting and, if any, the seconds it has left at most."""
        waiting = len(self.line) - 1
        return {
            "waiting": waiting,
            "turn_left": round(self.wait(1)) if waiting else None,
            "concurrency": self.concurrency(),
        }

    async def play(self, socket, chunk: Chunk, clock: float) -> float:
        """Send the chunk's frames one per 1 / FPS s from `clock` (or now, if production fell behind)."""
        loop = asyncio.get_running_loop()
        clock = max(clock, loop.time())
        for i, jpeg in enumerate(chunk.frames):
            await asyncio.sleep(max(0.0, clock - loop.time()))
            header = {"i": i, "n": len(chunk.frames)}
            if i == 0:
                header |= chunk.info | {"latency_ms": 1000 * (time.perf_counter() - chunk.started)} | self.turn()
                if chunk.preview is not None:
                    self.preview, fresh, self.fresh = chunk.preview, self.fresh, asyncio.Event()
                    fresh.set()
            await asyncio.wait_for(socket.send_bytes(pack(header, jpeg)), timeout=5.0)
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
    for env in demo.envs:
        for index in range(min(len(demo.datasets[env]), demo.prompt_cache_size // len(demo.envs))):
            demo.new_prompt(env, index=index, seed=0)
    demo.reset()
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
        default=["driving"],
    )
    parser.add_argument("--denoise_steps", type=int, help=f"default: {DENOISE_STEPS}")
    parser.add_argument(
        "--context_noise",
        type=float,
        help="override all environments; default: driving/indoor 0.35, robot/human manipulation 0.20",
    )
    parser.add_argument(
        "--decoder_context_frames",
        type=int,
        help="experimental shorter decoder context in video frames; default: full trained window (16)",
    )
    parser.add_argument(
        "--fixed_settings",
        action="store_true",
        help="keep all sessions at the server's denoising and context-noise settings",
    )
    parser.add_argument(
        "--fp8", action="store_true", help="experimental Hopper FP8 Linear inference; changes precision"
    )
    parser.add_argument("--fp8_fast_accum", action="store_true", help="use faster FP8 accumulation (requires --fp8)")
    parser.add_argument(
        "--fp8_decoder_only",
        action="store_true",
        help="apply FP8 only to the autoencoder decoder; keep world models in BF16 (requires --fp8)",
    )
    parser.add_argument("--fused_swiglu", action="store_true", help="combine SwiGLU input projections for inference")
    parser.add_argument(
        "--fused_swiglu_world_model_only",
        action="store_true",
        help="with --fused_swiglu, fuse only world models and leave the decoder unchanged",
    )
    parser.add_argument(
        "--cache_decoder_prefix",
        action="store_true",
        help="cache the constant decoder temporal register prefix for the selected context length",
    )
    parser.add_argument(
        "--short_decoder_attention",
        action="store_true",
        help="use direct causal attention for decoder sequences with at most two keys",
    )
    parser.add_argument(
        "--tiny_decoder_attention",
        action="store_true",
        help="use BF16 tensor-core causal attention for full-history decoder sequences",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile, with CUDA graphs"
    )
    parser.add_argument("--turn_seconds", type=float, default=120.0, help="a turn's length, once someone waits")
    parser.add_argument("--idle_seconds", type=float, default=30.0, help="a turn ends after this long without input")
    parser.add_argument("--prompt_cache", type=int, default=6, help="CPU prompt contexts primed at startup; 0 disables")
    parser.add_argument("--max_active", type=int, default=1, help="independent active users sharing one GPU worker")
    parser.add_argument(
        "--preproject_conditioning",
        action="store_true",
        help="group world-model conditioning projections across denoising steps",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="max ready users per inference batch; 1 disables")
    parser.add_argument("--batch_wait_ms", type=float, default=2.0, help="maximum extra wait to collect a batch")
    parser.add_argument("--max_line", type=int, default=100, help="the most sessions waiting in line")
    parser.add_argument("--log", type=Path, help="append a JSON line per visit (anonymous ids) to this file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.max_active < 1 or args.max_line < 0 or args.prompt_cache < 0:
        parser.error("max_active must be positive; max_line and prompt_cache must be nonnegative")
    if args.denoise_steps is not None and not 1 <= args.denoise_steps <= 50:
        parser.error("denoise_steps must be between 1 and 50")
    if args.context_noise is not None and (not math.isfinite(args.context_noise) or not 0 <= args.context_noise <= 1):
        parser.error("context_noise must be finite and between 0 and 1")
    if args.decoder_context_frames is not None and args.decoder_context_frames < 1:
        parser.error("decoder_context_frames must be positive")
    if args.batch_size < 1 or not math.isfinite(args.batch_wait_ms) or args.batch_wait_ms < 0:
        parser.error("batch_size must be positive and batch_wait_ms must be finite and nonnegative")
    if args.fp8_fast_accum and not args.fp8:
        parser.error("fp8_fast_accum requires fp8")
    if args.fp8_decoder_only and not args.fp8:
        parser.error("fp8_decoder_only requires fp8")
    if args.fused_swiglu_world_model_only and not args.fused_swiglu:
        parser.error("fused_swiglu_world_model_only requires fused_swiglu")
    return args


def build(args: argparse.Namespace, checkpoints: dict | None = None) -> Demo:
    """The demo of the parsed arguments; `checkpoints` (environment -> path or file object) replaces --weights."""
    device = torch.device(args.device)
    if checkpoints is None:
        pt = {env: Path(args.weights) / env / f"{args.autoencoder}.pt" for env in args.envs}
        checkpoints = {env: path if path.is_file() else f"{args.weights}/{env}" for env, path in pt.items()}
    autoencoder, world_models = load_models(args.autoencoder, checkpoints, device)
    decoder_context_frames = StreamingDecoder.validate_context_frames(autoencoder.config, args.decoder_context_frames)
    if device.type == "cuda":  # the weights autocast runs linear layers with, cast once rather than every call
        for model in (autoencoder, *world_models.values()):
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    module.to(torch.bfloat16)
    if args.fp8:
        from util.fp8 import convert_fp8_linears

        # The prompt encoder stays BF16. Decoder-only mode also preserves WM precision.
        targets = [("decoder", autoencoder.decoder)]
        if not args.fp8_decoder_only:
            targets.extend(world_models.items())
        for name, model in targets:
            summary = convert_fp8_linears(model, use_fast_accum=args.fp8_fast_accum)
            print(f"FP8 {name}: {summary['converted']} Linear layers (small-row projections retain BF16)", flush=True)
    if args.fused_swiglu:
        from util.fused_swiglu import convert_fused_swiglu

        targets = list(world_models.items())
        if not args.fused_swiglu_world_model_only:
            targets.insert(0, ("decoder", autoencoder.decoder))
        for name, model in targets:
            summary = convert_fused_swiglu(model)
            print(f"Fused SwiGLU {name}: {summary['converted']} modules", flush=True)
    if args.short_decoder_attention:
        for block in (*autoencoder.decoder.blocks, *autoencoder.decoder.tail_blocks):
            block.time_attn.short_temporal_attention = True
        print("Enabled short decoder temporal attention", flush=True)
    if args.tiny_decoder_attention:
        for block in (*autoencoder.decoder.blocks, *autoencoder.decoder.tail_blocks):
            block.time_attn.tiny_temporal_attention = True
        print("Enabled full-history decoder temporal attention kernel", flush=True)
    if args.cache_decoder_prefix:
        autoencoder.decoder.prepare_time_register_cache()
        print("Cached constant decoder temporal-register prefix", flush=True)
    if args.data:
        data, manifests = parse_pairs(args.data), parse_pairs(args.manifest)
    else:  # six clips per environment, shipped with the world models
        clips = Path(args.weights) / "demo_clips"
        if not clips.is_dir():
            clips = (
                Path(snapshot_download(args.weights, allow_patterns=[f"demo_clips/{env}/*" for env in world_models]))
                / "demo_clips"
            )
        data = {env: clips / env for env in world_models}
        manifests = {env: clips / env / "manifest.txt" for env in world_models}
    resolution = autoencoder.config.resolution
    datasets = {
        env: EnvironmentDataset(env, data[env], args.split, CLIP_FRAMES, resolution, manifests.get(env))
        for env in world_models
    }
    if args.compile:  # CUDA graphs of static shapes: a context, a step and a cache update per world model, a window
        torch._dynamo.config.recompile_limit = 8 * len(world_models) * min(args.batch_size, args.max_active)
        compile = dict(mode="reduce-overhead", dynamic=False)
        for world_model in world_models.values():
            world_model.velocity = torch.compile(world_model.velocity, **compile)
            if args.preproject_conditioning:
                world_model.project_schedule_conditioning = torch.compile(
                    world_model.project_schedule_conditioning, **compile
                )
        autoencoder.decoder.flow_forward = torch.compile(autoencoder.decoder.flow_forward, **compile)
    for world_model in world_models.values():
        world_model.preproject_conditioning = args.preproject_conditioning
    denoise_steps = args.denoise_steps or DENOISE_STEPS[args.autoencoder]
    demo = Demo(
        autoencoder,
        args.autoencoder,
        world_models,
        datasets,
        device,
        denoise_steps,
        args.context_noise,
        args.turn_seconds,
        args.idle_seconds,
        args.max_line,
        args.log,
        args.prompt_cache,
        fixed_settings=args.fixed_settings,
        decoder_context_frames=decoder_context_frames,
    )
    demo.precision = ("fp8-fast" if args.fp8_fast_accum else "fp8") if args.fp8 else "bf16"
    demo.fp8_decoder_only = args.fp8_decoder_only
    demo.decoder_precision = demo.precision
    demo.world_model_precision = "bf16" if args.fp8_decoder_only else demo.precision
    if args.fp8_decoder_only:
        demo.precision = f"bf16-wm/{demo.decoder_precision}-decoder"
    demo.fused_swiglu = args.fused_swiglu
    demo.fused_swiglu_world_model_only = args.fused_swiglu_world_model_only
    demo.preproject_conditioning = args.preproject_conditioning
    demo.cache_decoder_prefix = args.cache_decoder_prefix
    demo.short_decoder_attention = args.short_decoder_attention
    demo.tiny_decoder_attention = args.tiny_decoder_attention
    return demo


def produce_batch(slots: list[Demo]) -> list:
    """Prepare independent sessions, group compatible states, and run each group on the GPU worker."""
    results, groups = [None] * len(slots), {}
    for i, slot in enumerate(slots):
        try:
            prepared = slot.prepare_step()
            rollout = slot.rollouts[prepared[0]]
            key = batch_key(rollout, slot.denoise_steps, slot.context_noise)
            groups.setdefault(key, []).append((i, slot, prepared, rollout))
        except Exception as error:
            results[i] = error
    decoder_groups = {}
    for group in groups.values():
        try:
            first = group[0][1]
            rollouts = [entry[3] for entry in group]
            actions = [entry[2][1] for entry in group]
            if len(groups) > 1:
                pending = prepare_batch(rollouts, actions, first.denoise_steps, first.context_noise)
                decoder_groups.setdefault(decoder_batch_key(pending), []).append((group, pending))
                continue
            videos = (
                [rollouts[0].step(actions[0], first.denoise_steps, first.context_noise)]
                if len(group) == 1
                else step_batch(rollouts, actions, first.denoise_steps, first.context_noise)
            )
            # nvJPEG accepts a list of images: one call for all users avoids
            # repeating its setup and synchronization for every session.
            encoded = to_jpeg(torch.cat(videos, dim=0)) if len(group) > 1 else None
            offset = 0
            for (i, slot, prepared, _), video in zip(group, videos, strict=True):
                frames = None if encoded is None else encoded[offset : offset + len(video)]
                results[i] = slot.finish_step(prepared, video, len(group), frames=frames)
                results[i].info["decoder_batch_size"] = len(group)
                offset += len(video)
        except Exception as error:
            for i, _, _, _ in group:
                results[i] = error
    for entries in decoder_groups.values():
        try:
            nested = decode_batches([pending for _, pending in entries])
            videos = [video for group_videos in nested for video in group_videos]
            encoded = to_jpeg(torch.cat(videos, dim=0))
            offset = 0
            for (group, _), group_videos in zip(entries, nested, strict=True):
                for (i, slot, prepared, _), video in zip(group, group_videos, strict=True):
                    frames = encoded[offset : offset + len(video)]
                    results[i] = slot.finish_step(prepared, video, len(group), frames=frames)
                    results[i].info["decoder_batch_size"] = len(videos)
                    offset += len(video)
        except Exception as error:
            for group, _ in entries:
                for i, _, _, _ in group:
                    results[i] = error
    # The scheduler releases the whole job together, after every group's JPEGs
    # and optional previews are ready. Include that wait in each user's timing.
    completed = time.perf_counter()
    for result in results:
        if isinstance(result, Chunk):
            result.info["compute_ms"] = 1000 * (completed - result.started)
    return results


class ConcurrentDemo:
    """Independent rollout slots sharing immutable weights and a single GPU executor.

    Each slot retains the existing FIFO admission/turn policy. GPU jobs are serialized
    to preserve CUDA graph safety; browser playback and input run concurrently.
    """

    def __init__(self, demo: Demo, max_active: int, batch_size: int = 1, batch_wait_ms: float = 2.0):
        self.batch_size = min(batch_size, max_active)
        self.batch_wait_ms = batch_wait_ms
        demo.max_active = max_active
        if self.batch_size > 1:
            demo.scheduler = BatchScheduler(demo.executor, produce_batch, self.batch_size, batch_wait_ms)
        self.slots = [demo]
        demo.pool = self
        for _ in range(max_active - 1):
            slot = copy.copy(demo)
            slot.commands = queue.SimpleQueue()
            slot.line = []
            slot.preview, slot.fresh = None, asyncio.Event()
            slot.reset()
            self.slots.append(slot)

    def warm_up_batches(self, steps: int = 4) -> dict[int, float]:
        """Compile each admitted batch width before accepting traffic, using private rollout state."""
        timings = {}
        if self.batch_size < 2:
            return timings
        for env in self.slots[0].envs:
            for slot in self.slots[: self.batch_size]:
                slot.env = env
                slot.new_prompt(env, 0, seed=0)
            for width in range(2, self.batch_size + 1):
                for _ in range(steps):
                    started = time.perf_counter()
                    results = produce_batch(self.slots[:width])
                    for result in results:
                        if isinstance(result, Exception):
                            raise result
                    timings[width] = 1000 * (time.perf_counter() - started)
                print(f"Warmed {env} batch {width}: {timings[width]:.1f} ms per generation step", flush=True)
        envs = self.slots[0].envs
        if len(envs) > 1:
            patterns = list(itertools.combinations(envs, 2))
            patterns.extend(tuple(envs[i % len(envs)] for i in range(width)) for width in range(3, self.batch_size + 1))
            for pattern in patterns:
                slots = self.slots[: len(pattern)]
                for slot, env in zip(slots, pattern, strict=True):
                    slot.env = env
                    slot.new_prompt(env, 0, seed=0)
                for _ in range(steps):
                    started = time.perf_counter()
                    for result in produce_batch(slots):
                        if isinstance(result, Exception):
                            raise result
                    elapsed = 1000 * (time.perf_counter() - started)
                print(f"Warmed shared decoder {', '.join(pattern)}: {elapsed:.1f} ms per generation step", flush=True)
        for slot in self.slots:
            slot.reset()
        return timings

    async def session(self, socket):
        # No await between choosing a slot and entering session: reservation is atomic.
        slot = min(self.slots, key=lambda candidate: len(candidate.line))
        await slot.session(socket)

    def idle_slot(self, source: Demo, socket) -> Demo | None:
        """Only the first waiter of a slot may move, preserving that slot's FIFO order."""
        if len(source.line) < 2 or source.line[1] is not socket:
            return None
        return next((slot for slot in self.slots if slot is not source and not slot.line), None)

    def move_waiter(self, source: Demo, socket) -> Demo | None:
        """Atomically transfer a queue reservation; the caller has stopped its receive watcher."""
        destination = self.idle_slot(source, socket)
        if destination is not None:
            source.line.remove(socket)
            destination.line.append(socket)
        return destination


def serve(demo: Demo, host: str, port: int, max_active: int = 1, pool: ConcurrentDemo | None = None) -> None:
    """Serve the page and its websocket (the demo extra: starlette, uvicorn, websockets)."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    pool = pool or ConcurrentDemo(demo, max_active)
    page = (HERE / "demo.html").read_text()

    async def index(request):
        return HTMLResponse(page)

    async def health(request):
        return JSONResponse(
            {
                "ready": True,
                "pid": os.getpid(),
                "environments": demo.envs,
                "max_active": len(pool.slots),
                "batch_size": pool.batch_size,
                "batch_wait_ms": pool.batch_wait_ms,
                "denoise_steps": demo.defaults[0],
                "context_noise_by_env": demo.context_noise_defaults.copy(),
                "context_noise_override": demo.defaults[1],
                "decoder_context_frames": demo.decoder_context_frames,
                "cache_decoder_prefix": getattr(demo, "cache_decoder_prefix", False),
                "short_decoder_attention": getattr(demo, "short_decoder_attention", False),
                "tiny_decoder_attention": getattr(demo, "tiny_decoder_attention", False),
                "fused_swiglu": getattr(demo, "fused_swiglu", False),
                "fused_swiglu_world_model_only": getattr(demo, "fused_swiglu_world_model_only", False),
                "preproject_conditioning": getattr(demo, "preproject_conditioning", False),
                "settings_locked": demo.fixed_settings,
                "device": str(demo.device),
                "precision": getattr(demo, "precision", "bf16"),
                "world_model_precision": getattr(demo, "world_model_precision", "bf16"),
                "decoder_precision": getattr(demo, "decoder_precision", "bf16"),
                "fp8_decoder_only": getattr(demo, "fp8_decoder_only", False),
                "active": sum(bool(slot.line) for slot in pool.slots),
                "waiting": sum(max(0, len(slot.line) - 1) for slot in pool.slots),
                "concurrency": demo.concurrency(),
            }
        )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        if demo.scheduler is not None:
            await demo.scheduler.close()
        demo.executor.shutdown(wait=True)

    async def websocket(socket):
        await socket.accept()
        try:
            await pool.session(socket)
        except WebSocketDisconnect:
            pass

    app = Starlette(
        routes=[Route("/", index), Route("/health", health), WebSocketRoute("/ws", websocket)], lifespan=lifespan
    )
    # JPEGs are already compressed; per-message deflate costs CPU for no useful bandwidth reduction.
    uvicorn.run(app, host=host, port=port, log_level="warning", ws_per_message_deflate=False)


def main() -> None:
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    demo = build(args)
    pool = ConcurrentDemo(demo, args.max_active, args.batch_size, args.batch_wait_ms)
    # In the worker thread, which runs all the GPU work: CUDA graphs are recorded per thread.
    for env, ms in demo.executor.submit(warm_up, demo).result().items():
        tau = demo.autoencoder.config.tau
        print(f"{env}: {ms:.0f} ms per step of {tau} frames = {1000 * tau / ms:.1f} fps (real time: {FPS})", flush=True)
    for width, ms in demo.executor.submit(pool.warm_up_batches).result().items():
        print(f"batch {width}: {ms:.0f} ms / {demo.autoencoder.config.tau} frames per user (warmup only)", flush=True)
    print(f"serving {args.autoencoder} ({', '.join(demo.envs)}) on http://{args.host}:{args.port}", flush=True)
    serve(demo, args.host, args.port, pool=pool)


if __name__ == "__main__":
    main()
