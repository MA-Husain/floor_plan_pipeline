# brynx floor-plan pipeline

Phone capture in, one whole-property floor plan out: rooms with dimensions and ceiling heights,
doors / openings / windows, adjacency, per-surface damage with metric extent, concealed-damage
flags with the rule that fired, scope line items, a 95 % interval on every measurement, JSON +
rendered plan. Three input tiers, one output contract, one command.

## Run it (fresh machine, < 15 min plus download time)

```bash
git clone <this repo> && cd floor_plan_pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_weights.py          # ~7 GB, once; afterwards everything runs offline

python run_capture.py <capture>          # writes outputs/<name>/plan.json + plan.png
```

`<capture>` is any of the three tiers - the tier is detected from what you pass:

| You pass | Tier | Device | What the pipeline uses |
|---|---|---|---|
| a Stray Scanner folder (`rgb.mp4`, `depth/`, `odometry.csv`) | **LiDAR** | iPhone Pro (12 Pro+) | LiDAR depth + ARKit poses |
| a `.mov` / `.mp4` walkthrough | **Video** | any iPhone 15+ | RGB only -> MapAnything metric depth + poses |
| a folder of room folders of photos (`Kitchen/*.HEIC`, ...) | **Photo** | any iPhone 15+ | RGB stills + EXIF focal -> MapAnything |

How to capture: [docs/CAPTURE_PROTOCOL.md](docs/CAPTURE_PROTOCOL.md) (one page, written for a non-engineer).
Which tier on which phone and how accurate each is: [docs/DEVICE_MATRIX.md](docs/DEVICE_MATRIX.md).

Options: `--no-damage`, `--no-detect` (skip fixture detection), `--drift auto|on|off`, `--live`
(ignore cached model outputs; by default each model's output is cached in `outputs/<name>/cache`
and replays deterministically), `--tier` (override detection), `--out`.

## Output (schema `brynx.plan/1.0`)

`plan.json`: `rooms[]` (polygon, `area_m2` +- ci95, `dimensions.length_u/length_v` +- ci95,
`ceiling_height` +- ci95 or `null` when the ceiling was not seen, type + reason, fixtures),
`openings[]` (door / opening / window, `width_m` +- ci95, rooms it joins), `adjacency`,
`damage.regions[]` (class, surface, room, metric area +- ci95, extent, height), `concealed_damage_flags[]`
(rule id + finding), `scope[]` (line items keyed to damage and surface), `drift` (ablation numbers),
`error_model` (the tier's uncertainty budget), `timing_s`.

## Layout

```
run_capture.py            one command per capture (all tiers)
brynx/
  capture.py              Stray Scanner loader (LiDAR tier)
  rgb.py                  video / photo tiers: MapAnything windows, Sim(3) chaining, gravity, ReconCapture
  cloud.py geometry.py    depth -> world points, floor / ceiling / Manhattan frame
  plan.py cells.py        wall faces, door-gap barrier, Manhattan cell complex, space segmentation
  spaces.py semantics.py  structural typing (corridor / closet / hall / stairs) + YOLO-World fixtures
  drift.py                plane-anchored pose graph, revisit-ghosting metric, on/off ablation
  lidar.py                the plan run (tier-agnostic despite the name) + error budget
  damage.py               surface tiles -> CLIP defect probe -> regions, rules R1-R6, scope
  render.py               architectural plan drawing
  priors.py               every structural threshold, with its justification
benchmark/                ground truth (gt/) and scorers (eval_spaces.py, eval_tape.py)
tools/                    ablation, repeatability, RGB-vs-LiDAR, ceiling map, damage-on-a-photo, atlases
docs/                     capture protocol, device matrix, benchmark report, fix loop, technical report
models/defect_probe.npz   the trained damage classifier head (21 KB)
```

## Reproduce the reported numbers

```bash
python run_capture.py c7d28f72c6                           # LiDAR multi-room (held out)
python run_capture.py 1a8384c3f6                           # LiDAR multi-room
python run_capture.py c00a170fe1                           # LiDAR single room
python run_capture.py data/home/video/IMG_9085.mov --out outputs/home_video
python run_capture.py data/home/photos --out outputs/home_photos
python benchmark/eval_spaces.py c7d28f72c6                 # space segmentation vs GT
python benchmark/eval_tape.py outputs/home_video/plan.json benchmark/gt/home.json
python tools/drift_ablation.py c7d28f72c6                  # drift on / off
python tools/repeatability.py c7d28f72c6 1a8384c3f6        # same flat, two captures
python tools/rgb_vs_lidar.py c7d28f72c6                    # video tier vs LiDAR on the same walk
python scripts/train_defect_probe.py                       # damage classifier (BD3 held-out scores)
```

## Models and data (all run locally, disclosed)

| Model / data | Use | Licence |
|---|---|---|
| MapAnything `facebook/map-anything-apache` | metric depth + poses for video / photos | Apache-2.0 |
| DINOv2 (hub code) | MapAnything encoder code | Apache-2.0 |
| CLIP ViT-L/14 (OpenAI) | damage-tile features + surface gate | MIT |
| BD3 building-defect dataset (HF `chandrabhuma/building_defect_vqa`) | trains the damage probe | CC-BY-4.0 |
| YOLO-World v2-x (Ultralytics) | fixtures for room typing | AGPL-3.0 |
