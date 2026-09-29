# Rethinking Video Autoencoders for Interactive World Models

**CADRE**: **CA**usal **D**iffusion **RE**presentation autoencoders

Adrien Ramanana Rahary<sup>1,2</sup>, Yutong Bai<sup>3</sup>, Nicolas Dufour<sup>1</sup>, Patrick Pérez<sup>1*</sup>,
David Picard<sup>2</sup>

<sup>1</sup>Kyutai &nbsp; <sup>2</sup>École Nationale des Ponts et Chaussées &nbsp; <sup>3</sup>BAIR, UC Berkeley &nbsp;
<sup>*</sup>work done while at Kyutai

[![Paper](https://img.shields.io/badge/arXiv-coming_soon-b31b1b?logo=arxiv&logoColor=white)](#citation)
[![Rollouts](https://img.shields.io/badge/Rollouts-wmoptimal.github.io-green?logo=googlechrome&logoColor=white)](https://wmoptimal.github.io)
[![Weights](https://img.shields.io/badge/🤗%20Hugging%20Face-CADRE-yellow)](#models-and-weights)
[![License](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

![CADRE teaser](assets/teaser.jpg)

A world model that people or agents control live has to react fast. Its video autoencoder sets how many tokens each
frame costs and how long each action waits to take effect, yet world models usually borrow the autoencoders of
text-to-video generators, which were built for offline generation and cannot be retrained. **CADRE** is an open
family of seven video autoencoders at different space-time compressions: a trained bottleneck on frozen DINOv3
features, and a causal diffusion decoder that self-distils to one or two steps. We train the same world model on each
of them in four environments (robot and human manipulation, driving and indoor navigation). CADRE covers most of the
latency-quality frontier, and on it world models run at 20 fps on one H100, reacting in under 230 ms.

This repository has the code to train the autoencoders, world models and inverse-dynamics probes, to evaluate
rollouts and reconstructions, and an interactive browser demo of the released world models.

<table>
  <tr>
    <td><img src="assets/rollout_robot_manipulation.webp" width="384" alt="Robot manipulation rollout"></td>
    <td><img src="assets/rollout_driving.webp" width="384" alt="Driving rollout"></td>
  </tr>
  <tr>
    <td><img src="assets/rollout_indoor_navigation.webp" width="384" alt="Indoor navigation rollout"></td>
    <td><img src="assets/rollout_human_manipulation.webp" width="384" alt="Human manipulation rollout"></td>
  </tr>
</table>

<sub>Every frame above is generated in real time (20 fps on one H100) by the released CADRE-2x16x16 world models,
interactively, from the keys held in each caption. A real 2 s clip only primes the model and is not shown.</sub>

## Contents

- [Installation](#installation)
- [Quick start: the interactive demo](#quick-start-the-interactive-demo)
- [Models and weights](#models-and-weights)
- [Data](#data)
- [Training and evaluation](#training-and-evaluation)
- [Paper to code](#paper-to-code)
- [Citation](#citation)
- [License](#license)

## Installation

We manage the environment with [uv](https://docs.astral.sh/uv/) (Python 3.12, torch 2.8, CUDA 12.8 wheels):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed yet
git clone https://github.com/AdrienRR/cadre-autoencoders-and-world-models && cd cadre-autoencoders-and-world-models
uv sync                    # training and evaluation
uv sync --extra demo       # + the interactive demo (starlette, uvicorn, websockets)
uv sync --extra prepare    # + the dataset converters (h5py, mcap, pyarrow)
```

Prefix every command with `uv run`. Training starts DINOv3 from Meta's `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
([facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), under the DINOv3 License). Its model code comes
from `torch.hub` at a pinned commit, so the first build of any model needs network access.

## Quick start: the interactive demo

The demo runs the four CADRE-2x16x16 world models from a browser on one GPU. It
loads all four, with the single autoencoder they share, and starts each rollout from a real 2 s clip of a dataset in
the [DATA.md](DATA.md) format (its `test` split). The page shows only the generated frames.

```sh
uv sync --extra demo
uv run python demo.py --autoencoder cadre_2x16x16 --weights <weights>/world_models \
    --data robot_manipulation=<root> driving=<root> indoor_navigation=<root> human_manipulation=<root>
```

Then open http://localhost:8000 (if the demo runs on another machine, forward the port first, e.g.
`ssh -N -L 8000:localhost:8000 <machine>`). Startup compiles CUDA graphs, which takes a few minutes (`--no-compile`
starts at once and runs about 2.5x slower). `--envs` loads fewer environments. The page shows each environment's
controls, and its sliders change the number of denoising steps and the context noise.

The demo downloads `kyutai/cadre-world-models-2x16x16` from the Hugging Face Hub on first use; `--weights` also
takes a local copy.

## Models and weights

The weights are on the Hugging Face Hub, in the [CADRE Autoencoders & World Models](https://huggingface.co/collections/kyutai/cadre-autoencoders-and-world-models-6abbaa26ed6191f7c975072d) collection. Each model is
safetensors plus a `config.json`, and every entry point takes a Hub repo id wherever it takes a checkpoint.

- **Autoencoders**, one repo per compression τ×s×s (τ temporal, s spatial), all trained at 288x512 and 20 fps:
  `kyutai/cadre-1x48x48`, `-2x32x32`, `-4x24x24`, `-8x16x16`, `-1x16x16`, `-2x16x16` and `-4x16x16`, the
  150k-step pretraining checkpoints used in the paper. The 2x32x32, 2x16x16, 4x16x16 and 8x16x16 cells also come
  trained further, to 400k steps in total (a learning-rate cooldown, then a multi-resolution fine-tune), as
  `kyutai/cadre-<cell>-multires`: better reconstructions, and usable at other resolutions. The
  world models were trained on the 150k autoencoders, so the continued ones are not drop-in replacements for them.
- **World models**, `kyutai/cadre-world-models-2x16x16`: the 1.2B-parameter world models trained on
  CADRE-2x16x16, with one subfolder per environment (`robot_manipulation`, `driving`, `indoor_navigation`,
  `human_manipulation`) and the autoencoder they share.
- **Action probes**, `kyutai/cadre-action-probes`: the inverse-dynamics probes that score action following (ARR),
  one subfolder per environment.

```python
from model_autoencoder import CADRE
from model_idm import load_pretrained_probe
from model_world_model import load_pretrained

autoencoder = CADRE.from_pretrained("kyutai/cadre-2x16x16")
world_model, autoencoder = load_pretrained("kyutai/cadre-world-models-2x16x16", subfolder="driving", device="cuda")
probe = load_pretrained_probe("kyutai/cadre-action-probes", subfolder="driving")
```

The weights are released under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Every autoencoder
and world model contains the frozen DINOv3 ViT-B/16 encoder, which stays under Meta's
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Its agreement ships with the
weights.

## Data

The four environments are public datasets that we do not redistribute:
[ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) (robot manipulation),
[EgoDex](https://github.com/apple/ml-egodex) (human manipulation),
[PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (driving) and
[RealEstate10K](https://huggingface.co/datasets/chenchenshi/re10k-sc) (indoor navigation).
[DATA.md, "Preparing the data"](DATA.md#preparing-the-data) walks through downloading each one and converting it with
`prepare/<environment>.py`. Their defaults reproduce our preprocessing exactly. Human manipulation also needs the MANO hand
model: register at [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de), accept its licence and download "Models & Code"
(`mano_v1_2.zip`); `prepare/human_manipulation.py --mano_dir <mano_v1_2>/models` then fits it to EgoDex's hand joints. Check a conversion against ours with:

```sh
uv run --extra prepare python prepare/check.py --env <environment> --root <root>   # MATCH or MISMATCH
```

Every script reads one generic layout, `videos/<key>.mp4` at 20 fps, optional `actions/<key>.npy` and
`splits/{train,val,test}.txt` ([DATA.md, "The format"](DATA.md#the-format)), so you can train on your own videos by
exporting them to it. The held-out splits and the evaluation windows are in [manifests/](manifests/README.md).

## Training and evaluation

Every run is `torchrun` over a flat YAML recipe of `configs/`, validated by pydantic. `--batch_size` is global, so a
run on fewer GPUs keeps the recipe's batch as long as it divides evenly. Relaunching with the same `--output_dir`
resumes from the latest checkpoint.

### Train an autoencoder

Pick a cell, `cadre_<τ>x<s>x<s>` (`--help` lists them). `--data` takes one or more dataset roots, mixed by
`--data_weights`. The paper's autoencoders trained on a 23-source mixture (paper appendix). Every stage keeps a global
batch of 96 clips of 16 frames.

**Pretraining** (the paper's autoencoders): 150k steps at lr 2e-4 after a 2k-step warmup, on 16 GPUs (2 nodes x 8).

```sh
uv run torchrun --nnodes 2 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/pretrain.yaml --data <root> [<root> ...] [--data_weights <w> ...] \
    --dino_weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --output_dir runs/cadre_2x16x16
```

**Continued training** (the `-multires` autoencoders, not in the paper): two more stages give better reconstructions
and an autoencoder that handles several resolutions. The released ones ran them on 32 GPUs (4 nodes x 8, 3 clips each).

```sh
# Cooldown: continue to 360k steps, with a cosine decay to 2e-5 over the last 70k.
uv run torchrun --nnodes 4 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/cooldown_360k.yaml --data <root> ... \
    --continue_from runs/cadre_2x16x16/checkpoint-149999 --output_dir runs/cadre_2x16x16_360k

# Multi-resolution fine-tune: to 400k steps at 2e-5, on batches from 256x448 to 384x640.
uv run torchrun --nnodes 4 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/multires_ft.yaml --data <root> ... \
    --continue_from runs/cadre_2x16x16_360k/checkpoint-359999 --output_dir runs/cadre_2x16x16_multires
```

`--finetune_from <checkpoint or repo id>` starts from released weights at step 0.

### Train a world model

`--env` picks a recipe of `configs/world_model/`. The world model trains on a frozen autoencoder's latents, and its
checkpoint embeds that autoencoder.

```sh
# 100k steps, global batch 64 clips of 80 frames on 16 GPUs (2 nodes x 8 H100), lr 1e-4, AdamW (0.9, 0.99), wd 0.1.
uv run torchrun --nnodes 2 --nproc_per_node 8 main_world_model.py --env robot_manipulation \
    --autoencoder runs/cadre_2x16x16/checkpoint.pt --data <robot manipulation root> \
    --output_dir runs/wm_robot_manipulation_2x16x16
```

The four recipes differ only in the environment, its action statistics and the noise shift. One recipe serves every
autoencoder.

### Train an IDM probe

The probe reads actions back from 8-frame windows on frozen DINOv3 features. It trains on the world model's
dataset.

```sh
# 60k steps, 48 clips per GPU on 8 GPUs (batch 384), lr 3e-4 with cosine decay.
uv run torchrun --nproc_per_node 8 main_idm.py --env robot_manipulation --data <root> \
    --dino_weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --output_dir runs/idm_robot_manipulation
```

On fewer GPUs, raise `--grad_accum` to keep 384 clips per step.

### Evaluate

Rollouts: the manifest lists evaluation clips of 136 frames (a video and a start frame). For each one the world model
sees the first 40 real frames with their actions, then generates the next 96 under the recorded actions (10 denoising
steps, context noise 0.2). FID, FVD, PSNR and LPIPS are scored every 16 frames,
and ARR too when you pass a probe. Each clip's noise is seeded from its name, so any number of GPUs gives the same rollouts.

```sh
uv run torchrun --nproc_per_node 8 main_eval.py rollout --env robot_manipulation \
    --checkpoint <world model> --probe <probe> --data <root> \
    --manifest manifests/robot_manipulation_test.txt [--set decode_steps=2]
```

Reconstruction, on 3072 held-out 16-frame clips of [OpenVid-HD](https://huggingface.co/datasets/nkp37/OpenVid-1M)
(the `OpenVidHD` part of OpenVid-1M), decoded in one step:

```sh
uv run torchrun --nproc_per_node 8 main_eval.py reconstruction --checkpoint <autoencoder> \
    --data <OpenVid-HD root> --manifest manifests/openvid_hd_test.txt
```

Rank 0 prints the paper's numbers and writes every metric to `--output` (JSON). The paper reports one decoding step
and two (`--set decode_steps=2`). Human-manipulation ARR is read on the 12 wrist dimensions, indoor PSNR at frame 48.
Four robot-manipulation tasks have left the public ABC release, so `robot_manipulation_test.txt` has 2040 of the
paper's 2048 windows.

## Paper to code

- Compression τ×s×s and the seven cells ("What compression controls"): `CADRE_MODELS` in `model_autoencoder.py`.
- Frozen DINOv3 features and the 3D-conv bottleneck ("Architecture"): `model_autoencoder.py`, the encoder.
- Causal diffusion decoder, its three conditioning paths and streaming decode ("Architecture"): `model_autoencoder.py`,
  the decoder; `StreamingDecoder` in `model_world_model.py`.
- Flow-matching loss, L1 / LPIPS / DINO / 3D-wavelet terms, gradient balancing ("Training", Eq. 2): `losses.py`.
- Progressive self-distillation, one- or few-step decoding ("Training", Eq. 3): `losses.py` (`loss_psd`), `engine_autoencoder.py`.
- Autoencoder training, the cooldown and the multi-resolution fine-tune: `engine_autoencoder.py`,
  `configs/autoencoder/`.
- World model, a causal DiT with diffusion forcing, per-frame actions and the noise shift ("Judging an autoencoder by its world model", appendix):
  `model_world_model.py`, `engine_world_model.py`, `configs/world_model/`.
- Environments and their actions (Table 1): `ENVIRONMENTS` in `dataset.py`, `prepare/`, [DATA.md](DATA.md).
- Rollout protocol and streaming inference loop (appendix, "Inference"): `engine_eval.py`, `configs/eval/rollout.yaml`.
- FID, FVD, PSNR, LPIPS and ARR ("Judging an autoencoder by its world model"): `metrics.py`.
- Inverse-dynamics probe (appendix): `model_idm.py`, `engine_idm.py`.
- Reconstruction evaluation (appendix): `engine_eval.py`, `configs/eval/reconstruction.yaml`.
- Interactive demo: `demo.py`, `demo.html`.

## Citation

```bibtex
@misc{ramananarahary2026cadre,
  title     = {Rethinking Video Autoencoders for Interactive World Models},
  author    = {Ramanana Rahary, Adrien and Bai, Yutong and Dufour, Nicolas and P{\'e}rez, Patrick and Picard, David},
  year      = {2026},
  note      = {arXiv link coming soon}
}
```

## License

The code is under the [MIT License](LICENSE). The weights are under
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The DINOv3 code (loaded from its repository, not
vendored) and the frozen DINOv3 weights inside the models are under Meta's DINOv3 License. The datasets keep their own
licences (EgoDex: CC-BY-NC-ND; MANO: Max Planck non-commercial).

## Acknowledgements

CADRE builds on [DINOv3](https://github.com/facebookresearch/dinov3),
[RAE](https://github.com/bytetriper/RAE), [MIRA](https://github.com/mira-wm/mira), [LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[pytorch-fid](https://github.com/mseitzer/pytorch-fid) and [cd-fvd](https://github.com/songweig/content-debiased-fvd),
and on the [ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k), [EgoDex](https://github.com/apple/ml-egodex),
[PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles),
[RealEstate10K](https://huggingface.co/datasets/chenchenshi/re10k-sc) and
[OpenVid-1M](https://huggingface.co/datasets/nkp37/OpenVid-1M) datasets.
