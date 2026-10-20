"""Optional microbatch inference for independent demo rollouts.

The scheduler chooses compatible sessions and owns serialization of GPU work. This
module deliberately does not import ``demo`` so benchmark tools can use it too.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _layout(tensor: Tensor) -> tuple:
    # Concatenation copies every model input into a fresh batch. Source strides
    # can differ (e.g. a prompt window slice versus a later cloned window).
    return tensor.shape, tensor.dtype, tensor.device


def batch_key(rollout: Any, denoise_steps: int, context_noise: float) -> tuple:
    """Compatibility key for a single-session rollout, including its KV shapes.

    Temporal KV tensors flatten batch and spatial positions into their first axis;
    matching just their time dimension would incorrectly group different grids.
    """
    decoder = rollout.decoder
    return (
        id(rollout.world_model),
        id(rollout.autoencoder),
        id(decoder.autoencoder),
        rollout.device,
        rollout.tau,
        denoise_steps,
        context_noise,
        decoder.decode_steps,
        decoder.alpha,
        decoder.context_frames,
        _layout(decoder.window),
        _layout(decoder.noise),
        tuple(None if pair is None else tuple(_layout(tensor) for tensor in pair) for pair in rollout.kv),
    )


@dataclass
class PreparedBatch:
    """Owned state awaiting a shared decoder call; rollouts are committed only on success."""

    rollouts: tuple[Any, ...]
    last_actions: tuple[np.ndarray, ...]
    window: Tensor
    noise: Tensor
    windows: list[Tensor]
    noises: list[Tensor]
    caches: list[list]
    positions: tuple[tuple[int, int], ...]
    consumed: bool = False


def decoder_batch_key(prepared: PreparedBatch) -> tuple:
    """Shared-AE compatibility, independent of WM weights, denoising and action layout."""
    first = prepared.rollouts[0]
    decoder = first.decoder
    return (
        id(decoder.autoencoder),
        first.device,
        first.tau,
        decoder.decode_steps,
        decoder.context_frames,
        (prepared.window.shape[1:], prepared.window.dtype, prepared.window.device),
        (prepared.noise.shape[1:], prepared.noise.dtype, prepared.noise.device),
    )


@dataclass
class WorldModelInputs:
    rollouts: tuple[Any, ...]
    actions: tuple[np.ndarray, ...]
    action_embedding: Tensor
    noise: Tensor
    renoise: Tensor
    cache: list
    positions: tuple[tuple[int, int], ...]


@torch.no_grad()
def _stage_batch(rollouts, actions, denoise_steps, context_noise) -> WorldModelInputs:
    if len(rollouts) != len(actions):
        raise ValueError("each rollout needs one action array")
    if not rollouts:
        raise ValueError("a prepared batch must contain at least one rollout")
    if len({id(rollout) for rollout in rollouts}) != len(rollouts):
        raise ValueError("a rollout may occur only once in a batch")
    first = rollouts[0]
    expected_key = batch_key(first, denoise_steps, context_noise)
    if any(batch_key(rollout, denoise_steps, context_noise) != expected_key for rollout in rollouts[1:]):
        raise ValueError("rollouts have incompatible models, settings, or tensor layouts")
    if first.decoder.window.shape[0] != 1 or first.decoder.noise.shape[0] != 1:
        raise ValueError("each rollout must contain exactly one session")
    action_shape = (first.tau, first.world_model.config.action_dim)
    if any(not isinstance(action, np.ndarray) or action.shape != action_shape for action in actions):
        raise ValueError(f"actions must be numpy arrays with shape {action_shape}")
    if any(action.dtype != actions[0].dtype for action in actions[1:]):
        raise ValueError("action arrays in a batch must share a dtype")

    device = first.device
    decoder = first.decoder
    wm = first.world_model
    torch.compiler.cudagraph_mark_step_begin()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        a = wm.action_encoder(torch.from_numpy(np.stack(actions)).to(device))[:, 1:]
        shape, dtype = (1, 1, *decoder.window.shape[2:]), decoder.window.dtype
        # Each rollout owns its generator, so interleaving sessions changes no RNG
        # stream; drawing one larger random tensor would change every stream.
        draws = [(rollout.randn(shape, dtype), rollout.randn(shape, dtype)) for rollout in rollouts]
        noise, renoise = (torch.cat([draw[i] for draw in draws], dim=0) for i in range(2))
        kv = [
            None
            if pair is None
            else tuple(torch.cat([rollout.kv[layer][i] for rollout in rollouts], dim=0) for i in range(2))
            for layer, pair in enumerate(first.kv)
        ]
    return WorldModelInputs(
        tuple(rollouts),
        tuple(actions),
        a,
        noise,
        renoise,
        kv,
        tuple((rollout.position, rollout.steps) for rollout in rollouts),
    )


@torch.no_grad()
def _complete_batch(staged: WorldModelInputs, output) -> PreparedBatch:
    rollouts, actions = staged.rollouts, staged.actions
    first = rollouts[0]
    device = first.device
    x, new_kv = output
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        window = torch.cat([torch.cat([rollout.decoder.window[:, 1:] for rollout in rollouts]), x], dim=1)
        decoder_noise = torch.cat(
            [
                rollout.decoder.alpha * rollout.decoder.noise
                + (1 - rollout.decoder.alpha**2) ** 0.5 * rollout.decoder._randn()
                for rollout in rollouts
            ]
        )
        # Save session state before calling the decoder: compiled functions can
        # reuse graph-owned output buffers on their next invocation.
        windows = [window[i : i + 1].clone() for i in range(len(rollouts))]
        noises = [decoder_noise[i : i + 1].clone() for i in range(len(rollouts))]
        caches = [
            [
                None
                if pair is None
                else tuple(tensor[i * old[0].shape[0] : (i + 1) * old[0].shape[0]].clone() for tensor in pair)
                for pair, old in zip(new_kv, first.kv, strict=True)
            ]
            for i in range(len(rollouts))
        ]
    return PreparedBatch(
        tuple(rollouts),
        tuple(action[-1].copy() for action in actions),
        window,
        decoder_noise,
        windows,
        noises,
        caches,
        staged.positions,
    )


@torch.no_grad()
def prepare_batch(
    rollouts: Sequence[Any], actions: Sequence[np.ndarray], denoise_steps: int, context_noise: float
) -> PreparedBatch:
    """Prepare independent pending state, preserving serial RNG order and commit semantics."""
    staged = _stage_batch(rollouts, actions, denoise_steps, context_noise)
    first = staged.rollouts[0]
    with torch.autocast(first.device.type, dtype=torch.bfloat16, enabled=first.device.type == "cuda"):
        output = first.world_model.next_latent(
            staged.cache, staged.action_embedding, staged.noise, staged.renoise, denoise_steps, context_noise
        )
    return _complete_batch(staged, output)


@torch.no_grad()
def decode_batches(prepared_batches: Sequence[PreparedBatch]) -> list[list[Tensor]]:
    """Decode compatible pending batches together, then commit and return videos by group.

    A single pending batch reuses its contiguous model inputs. Multiple batches
    may come from different world models, action encoders or denoising settings.
    Decoder failure leaves every rollout's KV/window/position uncommitted.
    """
    if not prepared_batches:
        return []
    for batch in prepared_batches:
        if batch.consumed:
            raise ValueError("a prepared batch may be decoded only once")
        if any(
            (rollout.position, rollout.steps) != position
            for rollout, position in zip(batch.rollouts, batch.positions, strict=True)
        ):
            raise ValueError("rollout state changed after its batch was prepared")
    first_batch = prepared_batches[0]
    expected = decoder_batch_key(first_batch)
    if any(decoder_batch_key(batch) != expected for batch in prepared_batches[1:]):
        raise ValueError("prepared batches have incompatible decoder settings or tensor layouts")
    rollouts = [rollout for batch in prepared_batches for rollout in batch.rollouts]
    if len({id(rollout) for rollout in rollouts}) != len(rollouts):
        raise ValueError("a rollout may occur only once in a shared decoder batch")
    first = rollouts[0]
    decoder = first.decoder
    # All retained buffers were allocated outside compiled functions and cloned
    # before another WM call; a later graph replay cannot invalidate them.
    window = first_batch.window if len(prepared_batches) == 1 else torch.cat([b.window for b in prepared_batches])
    decoder_noise = first_batch.noise if len(prepared_batches) == 1 else torch.cat([b.noise for b in prepared_batches])
    with torch.autocast(first.device.type, dtype=torch.bfloat16, enabled=first.device.type == "cuda"):
        decode_window = window[:, -(decoder.context_frames // first.tau) :]
        decode_noise = decoder_noise[:, -decode_window.shape[1] * first.tau :]
        z = decoder.autoencoder.unnormalize(decode_window.permute(0, 1, 4, 2, 3))
        decode_tail = getattr(decoder.autoencoder, "decode_tail", None)
        video = (
            decode_tail(z, decoder.decode_steps, first.tau, noise=decode_noise)
            if callable(decode_tail)
            else decoder.autoencoder.decode(z, decoder.decode_steps, noise=decode_noise)
        ) * 0.5 + 0.5
        video = (video[:, -first.tau :].float().clamp(0, 1) * 255).round().to(torch.uint8)
        videos = [video[i].clone() for i in range(len(rollouts))]

    nested, offset = [], 0
    for batch in prepared_batches:
        for i, rollout in enumerate(batch.rollouts):
            rollout.kv = batch.caches[i]
            rollout.decoder.window = batch.windows[i]
            rollout.decoder.noise = batch.noises[i]
            rollout.position += rollout.tau
            rollout.last_action = batch.last_actions[i]
            rollout.steps += 1
        nested.append(videos[offset : offset + len(batch.rollouts)])
        offset += len(batch.rollouts)
        batch.consumed = True
    return nested


def step_batch(
    rollouts: Sequence[Any], actions: Sequence[np.ndarray], denoise_steps: int, context_noise: float
) -> list[Tensor]:
    """Advance one compatible WM batch and return independent uint8 videos."""
    if not rollouts:
        if actions:
            raise ValueError("each rollout needs one action array")
        return []
    return decode_batches([prepare_batch(rollouts, actions, denoise_steps, context_noise)])[0]
