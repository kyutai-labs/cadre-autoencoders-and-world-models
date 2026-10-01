# Data format

## Preparing the data

The four environments of the paper are public datasets. `prepare/<environment>.py` converts each download into the
format below, exactly as our world models and IDM probes read it. **The scripts' defaults reproduce our
preprocessing exactly: change no option to match the paper.** That includes robot manipulation's `--reencode`, on by
default, which reproduces a lossy re-encode of our pipeline that the released models depend on ("The paper's
environments", below). Run the scripts in the pinned environment, from `cadre/`:

```
uv sync --extra prepare      # pins every library the conversion depends on (PyAV's FFmpeg and x264, NumPy, SciPy, ...)
```

One detail depends on a library version: the wrist rotations of human manipulation match ours bit for bit with SciPy
1.14 and within 1e-6 with the pinned SciPy 1.17 (its rotation conversions changed); `prepare/check.py` allows for it.

Every script takes `--src <download> --out <root> [--split train|val|test|all] [--workers N]` (default: all splits,
8 processes). It converts only the episodes of our split lists (`manifests/splits/<environment>/`) that the download
holds, writes `<root>/splits/`, and skips finished episodes when run again. Start with `--split test`, the split the
rollout evaluation reads, then check it against ours:

```
uv run --extra prepare python prepare/check.py --env <environment> --root <root>    # MATCH or MISMATCH
```

`check.py` compares the episode and frame counts and aggregate statistics of the actions with fingerprints of our
conversion (`prepare/fingerprints.json`: aggregates only, no data).

### Robot manipulation (ABC-130k)

1. Accept the terms of [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) (gated, Apache-2.0) with your
   Hugging Face account, then log in: `uv run --with huggingface_hub hf auth login`.
2. Download the episodes of a split, here `test` (291 episodes, 41 GB):
   ```
   uv run --with huggingface_hub python -c "
   from huggingface_hub import snapshot_download
   keys = [line.split()[0] for line in open('manifests/splits/robot_manipulation/test.txt')]
   snapshot_download('XDOF/ABC-130k', repo_type='dataset', local_dir='<abc>',
                     allow_patterns=[f'data/train/{key}/episode.mcap' for key in keys])"
   ```
   For every split, download `allow_patterns=['data/train/*']` instead (21.6 TB; the paper's episodes, those of the
   RealSense stations, are 16.3 TB of it). Layout: `<abc>/data/train/<task>/episode_<uuid>/episode.mcap`.
3. Convert: `uv run --extra prepare python prepare/robot_manipulation.py --src <abc> --out <root> --split test`.

Four of the paper's tasks have left the public release since (215 train, 1 val and 1 test episode, 8 of the 2048
evaluation windows; manifests/README.md); the script converts what the download holds.

### Driving (PhysicalAI-AV)

1. Accept the NVIDIA Autonomous Vehicle Dataset licence of
   [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
   (gated), then log in: `uv run --with huggingface_hub hf auth login`.
2. Download `clip_index.parquet`, then the front-wide camera and egomotion chunks that hold a split's clips, here
   `test` (1024 clips in 624 chunks: 814 GB of camera, 26 GB of egomotion):
   ```
   uv run --extra prepare --with huggingface_hub python -c "
   import pyarrow.parquet as pq
   from huggingface_hub import hf_hub_download, snapshot_download
   repo, root = 'nvidia/PhysicalAI-Autonomous-Vehicles', '<av>'
   index = pq.read_table(hf_hub_download(repo, 'clip_index.parquet', repo_type='dataset', local_dir=root)).to_pydict()
   chunk = dict(zip(index['clip_id'], index['chunk']))
   chunks = {chunk[line.split()[0]] for line in open('manifests/splits/driving/test.txt')}
   snapshot_download(repo, repo_type='dataset', local_dir=root, allow_patterns=
       [f'camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_{c:04d}.zip' for c in chunks]
       + [f'labels/egomotion/egomotion.chunk_{c:04d}.zip' for c in chunks])"
   ```
   For every split, read the three split lists (2510 chunks: 3.3 TB of camera, 103 GB of egomotion). Layout:
   `<av>/clip_index.parquet`, `<av>/camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_<NNNN>.zip`,
   `<av>/labels/egomotion/egomotion.chunk_<NNNN>.zip`.
3. Convert: `uv run --extra prepare python prepare/driving.py --src <av> --out <root> --split test`.

### Indoor navigation (RE10K)

1. [chenchenshi/re10k-sc](https://huggingface.co/datasets/chenchenshi/re10k-sc) is not gated (Apache-2.0).
2. Download `train/index.json` and the shards that hold a split's scenes, here `test` (145 scenes in 144 shards,
   15 GB):
   ```
   uv run --with huggingface_hub python -c "
   import json
   from huggingface_hub import hf_hub_download, snapshot_download
   repo, root = 'chenchenshi/re10k-sc', '<re10k>'
   shard = json.load(open(hf_hub_download(repo, 'train/index.json', repo_type='dataset', local_dir=root)))
   shards = {shard[line.split()[0]] for line in open('manifests/splits/indoor_navigation/test.txt')}
   snapshot_download(repo, repo_type='dataset', local_dir=root, allow_patterns=[f'train/{s}' for s in shards])"
   ```
   For every split, download `allow_patterns=['train/*']` (4866 shards, 517 GB). Layout: `<re10k>/train/index.json`,
   `<re10k>/train/<NNNNNN>.torch`.
3. Convert: `uv run --extra prepare python prepare/indoor_navigation.py --src <re10k> --out <root> --split test`.

### Human manipulation (EgoDex)

1. Download Apple's archives (CC-BY-NC-ND, no registration). Every split needs all six, since the episodes of each
   split are spread over them (1.86 TB; the official `test.zip` is not used). Keep them zipped: the script reads the
   episodes from the archives.
   ```
   for part in part1 part2 part3 part4 part5 extra; do
       curl -C - -o <egodex>/$part.zip https://ml-site.cdn-apple.com/datasets/egodex/$part.zip
   done
   ```
   Layout: `<egodex>/<part>.zip`, each holding `<part>/<task>/<index>.mp4` and `<part>/<task>/<index>.hdf5`.
2. Get the MANO hand model (below): the finger actions are a MANO fit to the HDF5's finger joints.
3. Convert, on a GPU node if you can: `uv run --extra prepare python prepare/human_manipulation.py --src <egodex>
   --out <root> --mano_dir <mano>/mano_v1_2/models --split test`.

#### The MANO model

The release cannot ship MANO (Max Planck non-commercial licence), so download it yourself:

1. Register at https://mano.is.tue.mpg.de and accept the MANO licence.
2. On its Download page, download **Models & Code** (`mano_v1_2.zip`, 175 MB) and unzip it into `<mano>`. The script
   reads two of its files: `<mano>/mano_v1_2/models/MANO_LEFT.pkl` and `<mano>/mano_v1_2/models/MANO_RIGHT.pkl`.
3. Pass that folder as `--mano_dir <mano>/mano_v1_2/models`. chumpy, which MANO's own code needs, is not needed: the
   script reads the pickles itself.
4. A GPU is recommended: the fit (250 Adam steps per frame and hand) makes a CPU-only run about 5 times slower. The
   script uses the GPU when there is one; several `--workers` share it well.

### Time and disk space

Measured on our cluster (Lustre, 16 CPU cores of an H100 node, 1 GPU for human manipulation); a run scales with
`--workers`.

| environment | download: test / all | output: test / all | time: test / all | hardware, throughput |
|---|---|---|---|---|
| robot_manipulation | 41 GB / 16.3 TB | 2.6 GB / 1.0 TB | 52 min / 350 h | 16 cores, 215 frames/s |
| driving | 840 GB / 3.4 TB | 5.9 GB / 1.4 TB | 20 min / 80 h | 16 cores, ~19 core-s per clip |
| indoor_navigation | 15 GB / 517 GB | 0.22 GB / 51 GB | 1 min / 2 h | 16 cores, ~1.4 core-s per scene |
| human_manipulation | 1.86 TB / 1.86 TB | 1.6 GB / 240 GB | 20 min / 50 h | 1 GPU + 16 cores, 6 workers, 325 frames/s |

## The format

The paper's datasets are public but not ours to redistribute, so the code reads one generic format, which you
produce from any source with your own export script. Every consumer reads it through `dataset.ClipDataset` (the
world models and the IDM probes through its subclass `EnvironmentDataset`, below):

| consumer | clips | actions |
|---|---|---|
| autoencoder training (`main_autoencoder.py`) | 16 frames | ignored |
| world-model training | 80 frames | yes |
| IDM probe (`main_idm.py`) | 8 frames | yes |
| rollout evaluation | 136 frames (40 context + 96 generated) at the (clip, start) windows of a manifest | yes |

## Layout

```
<root>/
  videos/<key>.mp4        one video per episode, 20 fps, any resolution
  actions/<key>.npy       optional: float32 (frames, D)
  splits/train.txt        the videos of each split, one "<key> <frames>" line per video
  splits/val.txt
  splits/test.txt
```

- **key**: a relative path without spaces, e.g. `kitchen_03/episode_000012`; its video is `videos/<key>.mp4`.
- **videos**: any codec FFmpeg decodes (H.264 is the safe choice). They must be constant-rate 20 fps: frame `j` is
  the one presented `j / 20` s after the start of the stream. `<frames>` in the split file is the number of frames.
- **actions**: a dataset has actions for every video or for none. Row `j` is the action taken at frame `j`, the one
  that produces frame `j + 1`, in raw units (the world models normalize them). There is one row per video frame.
  Each environment has a fixed number of columns (`dataset.ENVIRONMENTS`, below), all continuous values.
- **splits**: disjoint lists of whole episodes (never split one episode across two lists). The autoencoder trains
  on `train`; a world model trains on `train` and validates on `val`; the IDM probe trains on the world model's
  `train` and measures its ceiling on `val`; rollout manifests pick their clips from `test` (or `val`).
- **manifest** (rollout evaluation): a text file of `<key> <start frame>` lines; each clip is frames
  `start .. start + 135` of that video.

Convert a source video with, for example (the paper's four environments have their own scripts, `prepare/<env>.py`,
below):

```
ffmpeg -i in.mp4 -vf fps=20 -c:v libx264 -crf 18 -an videos/<key>.mp4
```

## What a clip looks like

`ClipDataset(root, split, num_frames, resolution=(288, 512), manifest=None, actions=True)[i]` is

```
{"video": uint8 (T, 3, H, W), "actions": float32 (T, D), "key": "<key>@<start>"}
```

(`actions` only if the dataset has them and `actions=True`). `ds[i]` is window `i`; `ds[video, start]` is any clip
of the split's `video`-th video. Without a manifest, the windows tile every video from frame 0 without overlap
(`start = 0, T, 2T, ...`), leaving `slack = frames mod T` frames past the last full window (evaluation reads these fixed
windows); with a manifest, they are its windows.

`EnvironmentDataset(environment, root, split, num_frames, resolution=(288, 512), manifest=None)` is the same dataset
for one of the four environments of `dataset.ENVIRONMENTS`, as the world models and the IDM probes read it: it
requires actions with the environment's `action_dim` columns and passes each clip's actions through the
environment's `clip_actions` (only `indoor_navigation` has one, below).

**Resizing.** A frame is resized, antialiased bilinear, by the larger of the two ratios `H / h` and `W / w`, so that
it covers `(H, W)`, then center-cropped: no black bars and no aspect distortion (a 4:3 video loses a band at its top
and bottom; a 16:9 one is only downscaled). A video stored at exactly `(H, W)` is not resized. For the paper's
autoencoder training, sources of 540p and more were resized with FFmpeg's scaler instead (the others with this same
resize). The autoencoder itself pads a clip of another aspect ratio to 16:9 rather than cropping it (`CADRE.preprocess`); every
training and evaluation clip is already 288x512 by then.

**Multi-resolution finetune.** Clips are loaded at the largest bucket, 384x640 (cover and crop as above), and
each training step downscales the whole batch to the next bucket of `256x448, 288x512, 320x544, 352x608, 384x640`
(step `k` uses bucket `k mod 5`, the same on every GPU; antialiased bilinear, rounded to uint8). Store the videos
for this stage at 384x640 or more, or they are upscaled.

**Sampling.** Training mixes the `--data` roots by their weights: each clip draws a root with probability
proportional to its weight, then the next clip of that root's current epoch (`dataset.MixtureSampler`). An epoch
visits every window once, in a random order dealt round-robin to the GPUs, with each video's tiling shifted by a
random phase, an integer uniform in `[0, slack]` redrawn every epoch: a 150-frame episode gives 80-frame clips
starting anywhere in frames 0 to 70. Order and phases are drawn from (seed, root, epoch), so a
resumed run continues the stream exactly.

## Decoding library

`dataset.py` decodes with PyAV (`pip install av`). Its wheels bundle the FFmpeg libraries (including libx264 and
libdav1d), so no system FFmpeg is needed (torchcodec, by contrast, links against a system FFmpeg of a matching major
version). Frames are sought by presentation time and decoded from the preceding keyframe.

## The paper's environments

### Download

| environment | release | what the paper used |
|---|---|---|
| `robot_manipulation` (ABC) | [XDOF/ABC-130k](https://huggingface.co/datasets/XDOF/ABC-130k) (gated) | `data/train/<task>/episode_<uuid>/episode.mcap` of the RealSense stations (`/top-camera`, 640x480): the top camera and the arm and gripper states (below) |
| `driving` (PhysicalAI-AV) | [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) (gated, NVIDIA AV dataset licence) | `clip_index.parquet` (clip ids, official splits), `camera/camera_front_wide_120fov/*.zip`, the online egomotion `labels/egomotion/*.zip` |
| `indoor_navigation` (RE10K) | [chenchenshi/re10k-sc](https://huggingface.co/datasets/chenchenshi/re10k-sc) | the pixelSplat shards `train/*.torch`: per scene its JPEG `images`, `cameras` (world-to-camera) and `timestamps` (µs) |
| `human_manipulation` (EgoDex) | [apple/ml-egodex](https://github.com/apple/ml-egodex) (CC-BY-NC-ND): `https://ml-site.cdn-apple.com/datasets/egodex/<part>.zip` for `part1` to `part5` and `extra` | `<part>/<task>/<index>.mp4` and `.hdf5`; the official `test.zip` was never used |

`prepare/<environment>.py` converts each release to this format, exactly as defined below (`--help` for the
options). The frames the paper trained and evaluated on, against the public releases:
- `robot_manipulation`: a lossy re-encode, a BUG of our data pipeline that the released models depend on. The paper
  decoded a conversion of the MCAP files whose top camera was re-encoded (FFmpeg's bicubic nv12 -> RGB, then libx264
  `ultrafast`, GOP 2, crf 30), so the released robot-manipulation models were trained and evaluated on those degraded
  frames. `prepare/robot_manipulation.py --reencode`, the default, applies the same re-encode, bit-exact, before the
  frame selection. `--no-reencode` keeps the MCAP's own, cleaner frames, which score the released models worse for the
  wrong reason: on 128 paper clips, the released 2x32x32 world model's FID@96 is 42.8 on the paper's frames, 45.0 with
  `--reencode` and 54.8 without; LPIPS 0.146 / 0.145 / 0.162; the probe's real-video ceiling 0.919 / 0.920 / 0.906.
  Keep it on to reproduce the paper or use the released weights; turn it off only to train new models on the
  original frames.
- `driving`: a re-encode. The paper encoded the 640x360 frames at crf 23, then resized them to 288x512; the prepare
  script encodes them once, at crf 18.
- `indoor_navigation`: the release's own frames: the shards' JPEG images, decoded as the paper decoded them.
- `human_manipulation`: the same frame-exact Lanczos 288x512 transcode of the release's mp4s.

How each environment is defined. Use the same conventions to run the released world models and IDM probes on your own
export. The first column is the environment's name in `dataset.ENVIRONMENTS`, the recipes (`configs/world_model/`)
and `main_idm.py --env`.

| environment | video | actions (D) |
|---|---|---|
| `robot_manipulation` (ABC) | the MCAP's `/top-camera` (RealSense stations, 640x480, 30 fps; source frame `i` is its `i`-th message in log-time order), through our pipeline's lossy re-encode (above; `--reencode`): frame `j` is source frame `floor(3 j / 2)`, **stretched** to 288x512 by an antialiased bilinear resize | 14: `actions[j] = s[r(j + 1)] - s[r(j)]` with `r(j) = (3 j + 1) // 2` (clipped to the episode), where `s[i]` is `[left 6 joints, left gripper, right 6 joints, right gripper]`: the `position` of `/left-arm-state`, `/left-ee-state`, `/right-arm-state` and `/right-ee-state`, each the sample nearest to the log time of source frame `i` (the earlier on a tie); trailing frames with a NaN state or command are dropped |
| `driving` (PhysicalAI-AV) | the front wide 120-degree camera (1920x1080, 30 fps): frame `j` is source frame `ceil(1.5 j)`, scaled bicubic to 640x360 (FFmpeg's `fps=20,scale=-2:360:flags=bicubic`); a clip of `n` source frames has `(n - 1) * 2 // 3 + 1` frames, 403 for most | 2: [speed m/s, curvature 1/m] of the online egomotion, interpolated linearly at `t_0 + 50 ms * j` (`t_0` the capture time of the first frame): the forward speed in the car's frame and the curvature |
| `indoor_navigation` (RE10K) | 640x360 frames (mostly 30 fps), resampled to 20 fps (below) | 6: the camera twist `[v, w]` from frame `j` to `j + 1`: `v` the velocity in SfM units / s, `w` the angular rate in rad/s, as float32 (below); each clip's `v` is normalized when loaded |
| `human_manipulation` (EgoDex) | the head camera (1920x1080, 30 fps): frame `j` is source frame `floor(3 j / 2)`, scaled Lanczos to 288x512 | 102: per-step deltas of both hands' poses, 51 per hand (left first): wrist translation (3) and rotation (3), 15 finger joint rotations (45); rotations as the axis-angle of `R_t^T R_{t+1}` (below) |

**Human manipulation actions.** Frame `i` of an episode's mp4 (30 fps) has one row in its HDF5. Per hand (left at
0, right at 51), the pose of frame `i` is 51 values: the wrist translation `/transforms/<side>Hand[i, :3, 3]` (3),
the axis-angle of its rotation `[:3, :3]` (3), and 15 finger-joint axis-angles (45, MANO's joint order: index,
middle, pinky, ring, thumb, 3 joints each). Row `j` of the 20 fps actions is the step from source frame
`floor(3 j / 2)` to source frame `floor(3 (j + 1) / 2)` (the last clamped to the episode's last frame): the
translation difference, and for the wrist and each finger joint the axis-angle of `R_a^T R_b` (never the difference
of two axis-angle vectors). A hand's 51 deltas are zeroed on a step where either pose is all zeros (no EgoDex frame
is, in practice), and the HDF5 `/confidences` are not used. The wrist dims (0-5 and 51-56, the 12 that ARR reads)
come straight from the HDF5 transforms; they match the paper's to the last bit only with its SciPy version (the
rotation conversions). The finger joints are not in the HDF5: they are a MANO hand (v1.2, `flat_hand_mean=False`, so
its mean pose included) fitted frame by frame to 21 keypoints of the HDF5 (the wrist and each finger's knuckle,
intermediate base, intermediate tip and tip) by 250 Adam steps (lr 0.04, shape regularized by 0.01).
`prepare/human_manipulation.py --mano_dir <mano_v1_2/models>` runs that fit; the MANO model files come from
https://mano.is.tue.mpg.de (Max Planck non-commercial licence).

**Indoor navigation actions.** Frame `j` of the 20 fps video is image `round(j * fps / 20)` of the scene (`fps` from
its timestamps: `1 / median(diff(t))`, 30 or 24), with that image's camera-to-world pose (OpenCV axes, the inverse of
RE10K's world-to-camera matrix) and timestamp `t`. The actions are the twists of the trajectory of those frames,
Savitzky-Golay smoothed over the whole video (window 7, order 2: the translation directly, the rotations entrywise
then projected back onto rotations), at the frames' actual time steps: `v` the displacement from frame `j` to `j + 1`
in the camera frame at `j`, over the time step, and `w` the axis-angle of `R_j^T R_{j+1}`, over the time step; the last
row repeats the one before (`camera_twists` in `prepare/indoor_navigation.py`).

`v` is left in SfM units, whose scale is arbitrary per scene: `EnvironmentDataset` divides each clip's `v` by the
median `|v|` of its `T - 1` steps (`dataset.normalize_speed`; 1 if that median is 0), for training and evaluation clips
alike. The paper's world models were trained on actions that differ from these in three ways. Each clip's poses were
smoothed on their own, so a clip's first and last 3 rows differ (the filter's edge fit), and with them the median
speed a little. A scene's clips were normalized by the median speed of the first clip drawn from the scene (in each
epoch): the same scale as here when the scene holds one clip only, as every test scene does, a different one
otherwise. And the last row repeated the clip's previous twist, where here it is the step to the next frame. On 128
evaluation windows that show the paper's own images, the two scales agree within 1% for 91% of them, and away from the
edge rows `w` agrees exactly and `v` up to that ratio. In data built by `prepare/indoor_navigation.py`, 544 of the 2048
evaluation windows (27%) show the paper's images; the others differ in some of them (manifests/README.md).

**Indoor navigation keys.** The world models take the twist x 100 and 10 binary keys thresholded from the normalized
float32 twist: `forward`/`back` `v_z > 0.25` / `< -0.25`, `left`/`right` `v_x < -0.25` / `> 0.25`, `up`/`down`
`v_y < -0.25` / `> 0.25` (y points down), `fast`/`slow` `|v| > 1.5` / `< 0.5`, `roll_left`/`roll_right`
`w_z < -0.15` / `> 0.15`. They derive both from the twist themselves (`model_world_model.camera_inputs`), so a dataset
stores the twist only.

The paper's autoencoders trained on a mixture of 23 sources (listed in the paper's appendix); give each its own root
and its weight in `--data_weights`.
