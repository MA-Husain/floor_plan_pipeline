# brynz floor-plan pipeline

Phone capture in, one whole-property floor plan out: rooms with dimensions and ceiling heights,
doors / openings / windows, adjacency, per-surface damage with metric extent, concealed-damage
flags with the rule that fired, scope line items, a 95 % interval on every measurement, JSON +
rendered plan. Three input tiers, one output contract, one command. Runs fully offline.

## Prerequisites (empty machine)

You need **git** and **Python 3.11-3.14** (3.12 recommended), ~15 GB free disk, 16 GB RAM.

| OS | Install |
|---|---|
| macOS (Apple Silicon recommended) | `xcode-select --install` (gives git), then `brew install python@3.12` (Homebrew: https://brew.sh) or the installer from https://www.python.org/downloads/ |
| Ubuntu / Debian | `sudo apt update && sudo apt install -y git python3.12 python3.12-venv` |
| Windows | install Git from https://git-scm.com and Python 3.12 from https://www.python.org/downloads/ (tick "Add python.exe to PATH"); use `venv\Scripts\activate` instead of `source venv/bin/activate` |

Hardware: Apple Silicon (MPS) or an NVIDIA GPU (CUDA) are used automatically; CPU-only works but the
video and photo tiers become several times slower. No other tools (ffmpeg, COLMAP, CUDA toolkit) are needed.

## Run it (clean machine: ~10 min of setup + the one-time ~8 GB download)

```bash
git clone https://github.com/MA-Husain/floor_plan_pipeline.git && cd floor_plan_pipeline
python3.12 -m venv venv && source venv/bin/activate      # any Python 3.11-3.14 (developed on 3.14, Apple Silicon)
pip install -r requirements.txt
python scripts/fetch_weights.py                            # once: model code + weights into ./third_party, ./weights

python run_capture.py <capture>                            # -> outputs/<name>/plan.json, plan.png, damage.png
```

`<capture>` can be any tier - the tier is detected from what you pass:

| You pass | Tier | Phone | What the pipeline uses |
|---|---|---|---|
| a Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`) | **LiDAR** | iPhone 12 Pro or newer Pro | LiDAR depth + ARKit poses |
| a `.MOV` / `.mp4` walkthrough | **Video** | any iPhone 15+ | RGB only: MoGe-2 metric depth + MapAnything poses |
| a folder of room folders (`flat/Kitchen/*.HEIC`, `flat/Bedroom1/*.HEIC`, ...) | **Photo** | any iPhone 15+ | RGB stills + EXIF focal: MoGe-2 depth, per-room MapAnything, rooms joined by doorway photos |

How to capture (one page, for a non-engineer): [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md).
Which tier on which phone, accuracy and run time: [docs/DEVICE_MATRIX.md](docs/DEVICE_MATRIX.md).

Options: `--no-damage`, `--no-detect` (skip fixture detection), `--drift auto|on|off`, `--live`
(ignore cached model outputs), `--rotate cw|ccw|180` (a sideways video, e.g. a Stray `rgb.mp4` run
as an RGB-only video), `--tier` (override detection), `--out`.

Run time on an Apple M5 / 16 GB: LiDAR ~7 min, photos ~7 min, video ~10-15 min per minute of video.

## Output (schema `brynz.plan/1.0`)

Three files per capture: `plan.png` (the floor plan; damaged walls marked in red D1, D2, ...),
`damage.png` (evidence: for each damage region the camera frame it was judged from, tiles boxed,
with class, room, area and the concealed-damage rule fired) and `plan.json` (every number). A damage
summary is also printed at the end of the run.


`plan.json`: `rooms[]` (polygon, `area_m2`, `dimensions.length_u/length_v` between the principal
walls and `extent_u/extent_v`, each +- `ci95`; `ceiling_height` +- `ci95`, or `null` when the
ceiling was not seen; type + reason; fixtures), `openings[]` (door / opening / window, `width_m`
+- `ci95`), `adjacency`, `damage.regions[]` (class, surface, room, area m2 +- `ci95`, extent,
height), `concealed_damage_flags[]` (rule + finding), `scope[]` (line items keyed to damage and
surface), `drift` (ablation numbers), `error_model` (the tier's uncertainty budget), `timing_s`.
Photo tier also: `position_known` per room and `stitch` (links, unplaced rooms, overlaps).

## Documents
| | |
|---|---|
| [docs/COMPLIANCE_MATRIX.md](docs/COMPLIANCE_MATRIX.md) | requirement -> file -> artifact -> status |
| [docs/BENCHMARK.md](docs/BENCHMARK.md) | gates per tier, repeatability, head-to-head, timing |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | architecture, tiers, drift, error budget, calibration, fix loop, failure modes |
| [docs/fixloop/](docs/fixloop/) | fix declaration, before / after runs, how to regenerate |
| [results/](results/) | snapshot of every plan, render, ablation and damage overlay reported |

## Layout
```
run_capture.py            one command per capture (all tiers)
brynz/
  capture.py              Stray Scanner loader (LiDAR tier)
  rgb.py                  video / photo tiers: keyframes, MapAnything windows, photo stitching, gravity
  mono.py                 MoGe-2 metric depth (scale source of the RGB tiers, calibrated vs LiDAR)
  photo.py                photo tier: per-room plans placed into one plan
  cloud.py geometry.py    depth -> world points, floor / ceiling / Manhattan frame
  plan.py cells.py        wall faces, door-gap barrier, Manhattan cell complex, space segmentation
  spaces.py semantics.py  structural typing (corridor / closet / hall / stairs) + YOLO-World fixtures
  drift.py                plane-anchored pose graph, revisit-ghosting metric, on/off ablation
  lidar.py                the plan run (tier-agnostic despite the name) + error budget
  damage.py               surface tiles -> CLIP defect probe -> regions, rules R1-R6, scope
  render.py               architectural plan drawing
  priors.py               every structural threshold, with its justification
benchmark/                ground truth (gt/) and scorers (eval_spaces.py, eval_tape.py)
tools/                    ablation, repeatability, RGB-vs-LiDAR, depth-scale eval, photo-set builder, diagnostics
models/defect_probe.npz   the trained damage classifier head (21 KB)
```

## Reproduce the reported numbers
Model outputs are cached in `outputs/<name>/cache/` and replay deterministically; `--live` re-runs
the models. Raw captures and caches are too large for git (see the submission email for the data link).

```bash
# LiDAR tier on the three sample captures
python run_capture.py c00a170fe1 && python run_capture.py 1a8384c3f6 && python run_capture.py c7d28f72c6
python benchmark/eval_spaces.py c7d28f72c6                     # space segmentation vs frame-level GT (held out)
python tools/drift_ablation.py c7d28f72c6                      # drift on / off
python tools/repeatability.py c7d28f72c6 1a8384c3f6 --rooms    # repeatability (fix-loop gate)
python tools/ceiling_view.py c7d28f72c6 ceiling.png            # where the ceiling was actually seen

# Video tier on the sample captures' own RGB (LiDAR as the reference)
python tools/depth_scale_eval.py                               # which model gives true metric scale
python tools/rgb_vs_lidar.py c00a170fe1                        # depth + trajectory vs LiDAR / ARKit
python run_capture.py c00a170fe1/rgb.mp4 --rotate cw --out outputs/c00a_videotier

# Photo tier: per-room stills cut from a LiDAR capture, LiDAR plan as the reference
python tools/make_photo_set.py c7d28f72c6 --per-room 5
python run_capture.py data/photo_sets/c7d28f72c6 --out outputs/c7_photos_v2

# Own iPhone 17 (non-Pro) data with tape ground truth + magicplan export
python run_capture.py data/home/video/IMG_9085.mov --out outputs/home_video
python run_capture.py data/home/photos --out outputs/home_photos_v2
python benchmark/eval_tape.py outputs/home_photos_v2/plan.json benchmark/gt/home.json

# Damage
python scripts/train_defect_probe.py                           # BD3 held-out scores -> models/defect_probe.json
python tools/damage_photo.py <photo> overlay.jpg               # damage check on a single photo
```

## Models and data (all run locally, disclosed)

| Model / data | Use | Licence |
|---|---|---|
| MoGe-2 ViT-L `Ruicheng/moge-2-vitl-normal` (Microsoft) | metric depth + focal for video / photos | MIT |
| MapAnything `facebook/map-anything-apache` (Meta) + DINOv2 hub code | camera poses for video / photos | Apache-2.0 |
| CLIP ViT-L/14 (OpenAI) | damage-tile features + bare-surface gate | MIT |
| BD3 building-defect dataset (HF `chandrabhuma/building_defect_vqa`) | trains the damage probe | CC-BY-4.0 |
| YOLO-World v2-x (Ultralytics) | fixtures for room typing | AGPL-3.0 |

Also evaluated and not used (numbers in the benchmark report): Depth-Anything-3 Metric (scale 0.93x),
COLMAP/GLOMAP via pycolmap (fragments these indoor videos), Grounding DINO + SAM 2.1 (missed texture
defects).
