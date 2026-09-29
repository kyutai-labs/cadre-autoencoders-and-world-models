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

The demo runs the four CADRE-2x16x16 world models from a browser on one GPU (H100: 20 fps at 7 denoising steps). It
loads all four, with the single autoencoder they share, and starts each rollout from a real 2 s clip of a dataset in
the [DATA.md](DATA.md) format (its `test` split). The page shows only the generated frames.

```sh
uv sync --extra demo
uv run python demo.py --autoencoder cadre_2x16x16 --weights <weights>/world_models \
    --data robot_manipulation=<root> driving=<root> indoor_navigation=<root> human_manipulation=<root>
```

Then open http://localhost:8000. On a cluster node, open a tunnel from your machine first:
`ssh -N -L 8000:<node>:8000 <login host>`. Startup compiles CUDA graphs, which takes a few minutes (`--no-compile`
starts at once and runs about 2.5x slower). `--envs` loads fewer environments. `--denoise_steps` and
`--context_noise` (defaults 7 and 0.15) can also be changed from the page.

The demo downloads `kyutai/cadre-world-models-2x16x16` from the Hugging Face Hub on first use; `--weights` also
takes a local copy.

**Controls.** The two hands mirror each other: left hand Q W E / A S D, right hand U I O / J K L.

- Every environment: `1`-`4` switch tabs (Manipulation (Robot), Manipulation (Human), Navigation (Driving),
  Navigation (Indoor)), `N` starts from a new prompt clip, `Space` toggles between interactive and replay (the clip's
  recorded actions, which stop at their end rather than looping).
- Manipulation (Robot), the 14 joints of two YAM arms: `A`/`D` turn the left base (J0), `Q`/`E` select the left
  arm's other joint (J1-J5 or the gripper), `W`/`S` move it. The right arm uses `J`/`L`, `U`/`O` and `I`/`K`.
- Manipulation (Human), both wrists of a 102-D hand pose: left wrist `A`/`D` sideways, `W`/`S` forward and back,
  `Q`/`E` down and up. The right wrist uses `J`/`L`, `I`/`K` and `U`/`O`. The fingers and wrist rotations hold
  still.
- Navigation (Driving), speed and curvature: `W` throttle, `S` brake, `X` hard brake, `A`/`D` steer. The planned path
  is drawn over the video and can be hidden with the "path" toggle.
- Navigation (Indoor), the 6-D camera twist: `W`/`S` forward and back, `A`/`D` strafe, `Q`/`E` down and up; look with
  the mouse (click the video to lock the pointer, `Esc` releases it) or with `J`/`L` yaw, `I`/`K` pitch, `U`/`O`
  roll. Hold `Shift` for fast, `C` for slow.

Every action stays inside the range the environment's training data covers.

## Models and weights

The weights are on the Hugging Face Hub, in the [CADRE Autoencoders & World Models](https://huggingface.co/collections/kyutai/cadre-autoencoders-and-world-models-6abbaa26ed6191f7c975072d) collection. Each model is
safetensors plus a `config.json`, and every entry point takes a Hub repo id wherever it takes a checkpoint.

- **Autoencoders**, one repo per compression τ×s×s (τ temporal, s spatial), all trained at 288x512 and 20 fps:
  `kyutai/cadre-1x48x48`, `-2x32x32`, `-4x24x24`, `-8x16x16`, `-1x16x16`, `-2x16x16` and `-4x16x16`, the
  150k-step pretraining checkpoints used in the paper. The 2x32x32, 2x16x16, 4x16x16 and 8x16x16 cells also come
  continued to 360k steps with a learning-rate cooldown, then a multi-resolution fine-tune, as
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
`prepare/<environment>.py`. Their defaults reproduce our preprocessing exactly. Human manipulation also needs the MANO
hand model, which you download yourself. Check a conversion against ours with:

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
`--data_weights`. The paper's autoencoders trained on a 23-source mixture (paper appendix).

```sh
# 1. Pretraining: 150k steps, global batch 96 on 16 GPUs (2 nodes x 8 H100), lr 2e-4 after a 2k-step warmup.
uv run torchrun --nnodes 2 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/pretrain.yaml --data <root> [<root> ...] [--data_weights <w> ...] \
    --dino_weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --output_dir runs/cadre_2x16x16

# 2. Cooldown: continue to 360k steps, with a cosine decay to 2e-5 over the last 70k steps.
uv run torchrun --nnodes 2 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/cooldown_360k.yaml --data <root> ... \
    --continue_from runs/cadre_2x16x16/checkpoint-149999 --output_dir runs/cadre_2x16x16_360k

# 3. Multi-resolution fine-tune: 12 h at 2e-5, on batches from 256x448 to 384x640.
uv run torchrun --nnodes 2 --nproc_per_node 8 main_autoencoder.py --model cadre_2x16x16 \
    --config configs/autoencoder/multires_ft.yaml --data <root> ... \
    --continue_from runs/cadre_2x16x16_360k/checkpoint-359999 --output_dir runs/cadre_2x16x16_ft
```

All three stages keep the global batch of 96 clips of 16 frames. The released continued autoencoders ran stages 2
and 3 on 32 GPUs (4 nodes, 3 clips each). `--finetune_from <checkpoint.pt>` starts from release weights at step 0.

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

On fewer GPUs, raise `--grad_accum` to keep 384 clips per step. The released probes are the step-55000 checkpoints
(human manipulation: step 60000).

### Evaluate

Rollouts: each window of the manifest gives 40 real context frames with their actions, then 96 generated frames (10
denoising steps, context noise 0.2) under the recorded actions. FID, FVD, PSNR and LPIPS are scored every 16 frames,
and ARR too when you pass a probe. Any number of GPUs gives the same rollouts.

```sh
uv run torchrun --nproc_per_node 8 main_eval.py rollout --env robot_manipulation \
    --checkpoint <world model> --probe <probe> --data <root> \
    --manifest manifests/robot_manipulation_test.txt [--set decode_steps=2]
```

Reconstruction, on the 3072 held-out OpenVid-HD windows, 16 frames each, decoded in one step:

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

## Results

CADRE covers most of the latency-quality frontier in all four environments, and its lead widens when the camera
moves. The study behind it comes to four findings. At strong compression, a diffusion decoder is sharper than
perceptual regression. In robot manipulation, temporal compression works better in the autoencoder than in the world
model. Manipulation and navigation call for opposite compressions. With a still camera, coarser spatial tokens hurt
motion far more than image quality. The paper has the full numbers, and [wmoptimal.github.io](https://wmoptimal.github.io)
has uncurated rollouts.

## Citation

```bibtex
@inproceedings{ramananarahary2027cadre,
  title     = {Rethinking Video Autoencoders for Interactive World Models},
  author    = {Ramanana Rahary, Adrien and Bai, Yutong and Dufour, Nicolas and P{\'e}rez, Patrick and Picard, David},
  booktitle = {Under review at the International Conference on Learning Representations (ICLR)},
  year      = {2027},
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
and on the ABC-130k, EgoDex, PhysicalAI-AV and RealEstate10K datasets.
