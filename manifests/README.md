# Evaluation manifests and splits

| file | what |
|---|---|
| `<env>_test.txt` | The rollout windows of each environment of `dataset.ENVIRONMENTS`: 2048 `<key> <start frame>` lines at 20 fps (2040 for robot_manipulation, below; the evaluation reads 136 frames from each start). |
| `openvid_hd_test.txt` | The reconstruction windows: 3072 16-frame windows of 1170 OpenVid-HD videos. |
| `splits/<env>/{train,val,test}.txt` | `<key> <frames>` split lists (DATA.md). Copy them to `<root>/splits/`. |

Download each environment's data from the links of DATA.md's Download table (`../DATA.md`, "The paper's
environments") and convert it with `prepare/<env>.py`. `main_eval.py` reads a manifest's videos from the `test` split. A key is the video's file stem, `videos/<key>.mp4`,
built from the public dataset's own identifiers:

| env | key | public identifier |
|---|---|---|
| robot_manipulation (ABC) | `<task>/episode_<uuid>` | the episode's directory in XDOF/ABC-130k, `data/train/<task>/episode_<uuid>/episode.mcap`; its `/top-camera` stream |
| driving (PhysicalAI-AV) | `<clip_id>` | PhysicalAI-AV clip UUID; `camera_front_wide_120fov` |
| indoor_navigation (RE10K) | `<scene>` | RealEstate10K scene id (the name of its camera file), official train split |
| human_manipulation (EgoDex) | `<part>/<task>/<index>` | the episode's path in Apple's `<part>.zip` (`<index>.mp4`, `<index>.hdf5`) |
| openvid_hd | `<clip>` | OpenVid-HD clip file stem (validation parquet shards) |

ABC keys: the paper read a conversion of the MCAP files that numbered each task's episodes in the `sorted()` order of
their directories, one camera station at a time, minus 4 undecodable episodes. The keys here are the episode
directories that numbering maps back to, each checked against its MCAP's `/top-camera` frame count.

`<frames>` counts 20 fps frames: 2/3 of the 30 fps frames, rounded down (ABC, EgoDex, OpenVid-HD); `(n - 1) * 2 // 3
+ 1` of the `n` 30 fps frames for driving (403 for most clips); for RE10K, the frames j whose image
`round(j * fps / 20)` exists (fps from the scene's timestamps, 30 or 24).

## Where the windows come from

These are the windows the paper scored. Its evaluation drew each GPU's clips from the held-out split: windows at a
random offset per video, shuffled, at most 8 per video (driving: 2, all of them), 2048 in total. The draw therefore
depended on the number of GPUs, which was 8 for robot manipulation, 4 for driving and indoor navigation and 2 for
human manipulation; the manifests list the windows of that draw. The robot-manipulation CADRE-1x16x16 world model is
the one exception: it was scored on 4 GPUs, on a disjoint set of 2048 windows.

A start is the 20 fps frame nearest the first frame the paper decoded: exact for driving (a 20 fps source), within one
source frame for the 30 fps sources (ABC, EgoDex; RE10K images, resampled by rounding, as above). The RE10K draw
repeats windows: its 2048 lines hold 1702 distinct windows; 41 lines that the rounding pushed one frame past the end
of their scene start one frame earlier. The paper resampled each RE10K window from its own first image, the data here
resamples the whole scene from its first image: 1344 of the 2048 windows start on the paper's first image, and 544
show the paper's images throughout.

OpenVid-HD: the videos of the validation parquet shards, re-encoded at 30 fps (libx264, crf 18), with windows at 0,
0.8 and 1.6 s.

## Splits

The paper had one held-out split per environment. `test` holds its videos with an evaluation window, `val` the other
held-out videos, `train` what the world models and the IDM probes trained on.

- robot_manipulation: 1% of the episodes, `md5("<task>/<i>") % 100 < 1` with `i` the episode's index in our LeRobot
  conversion (so only these lists reproduce it). The paper used the RealSense stations only (`/top-camera`); four of
  its tasks (`dust_the_bookshelf_with_a_duster`, `mix_the_coffee_capsules_in_the_storage_basket`,
  `put_the_gloves_into_the_storage_basket`, `tie_the_trash_bag`) are no longer in the public release, so these lists
  leave out their 215 train, 1 val and 1 test episodes and their 8 evaluation windows: `robot_manipulation_test.txt`
  has 2040 windows, where the paper's scores used 2048.
- driving: the official PhysicalAI-AV split; `test` and `val` are its val clips; its official test was never used.
- indoor_navigation: 1% of the official train scenes, by the md5 of their path on our cluster (so only these lists
  reproduce it); a window needs 7.2 s, which leaves 145 held-out scenes to evaluate on.
- human_manipulation: 1% of the episodes of parts 1-5 and extra, `md5("<part>/<task>/<index>") % 100 < 1`; the
  official test set was never used.

## Actions beyond DATA.md

The paper computed each window's actions on the fly, so a per-video `actions/<key>.npy` reproduces them only
approximately:

- robot_manipulation: state rows `round(30 t)` against frames `floor(30 t)` (DATA.md's formula), one row off on some
  steps.
- driving: rows `round(20 t)` against frames `floor(20 t)`: for about half of the windows the paper's commands led
  the frames by one step; the release aligns row j with frame j.
- indoor_navigation: the data stores the raw twist of the whole video and each clip's velocity is normalized by its
  own median speed (DATA.md). The paper smoothed each window's poses on their own and normalized by the first window
  drawn from the scene. Each of the 145 test scenes hosts one window only, so the paper's scale was the window's own
  too: on 128 windows that show the paper's images (above), the two scales agree within 1% for 91% of them and within
  5% for all, `w` agrees exactly away from the window's first and last 3 rows, and those edge rows differ (the
  smoothing).
- human_manipulation: the 12 wrist dims (the ones ARR reads) come from the HDF5 hand transforms; the 90 finger dims
  from MANO fits to the HDF5 finger joints (`prepare/human_manipulation.py --mano_dir`, DATA.md).
