# Benchmark report

All numbers regenerate from raw inputs with the commands shown (cached model outputs replay
deterministically; `--live` re-runs the models). Machine: Apple M5, 16 GB, MPS, offline.

## Benchmark set

| Capture | What | Tier(s) run | Ground truth |
|---|---|---|---|
| `c00a170fe1` (sample) | single room + bathroom + living, 38 s | LiDAR, video (its RGB only), photo | LiDAR plan (same walk) |
| `1a8384c3f6` (sample) | one floor, 7 spaces + connector | LiDAR, video, photo | frame-level space labels `benchmark/gt/1a8384c3f6.json`; LiDAR |
| `c7d28f72c6` (sample, **held out** - never tuned on) | same floor, with ceiling sweeps | LiDAR, video, photo | `benchmark/gt/c7d28f72c6.json`; LiDAR |
| `data/home` (own, iPhone 17 non-Pro) | 3-bed flat, 10 room folders, 165 s video, real damage in Room3 | video, photo | tape: Room2 2.718 m (ceiling) x 3.708 x 3.048 m |

`1a8384c3f6` and `c7d28f72c6` are the same flat captured twice -> repeatability. The video and photo
tiers on the sample captures use only the RGB video / stills cut from it (`tools/make_photo_set.py`):
depth and poses are never read, the LiDAR plan of the same walk is the reference.

## LiDAR tier

**Space segmentation** (`python benchmark/eval_spaces.py <capture>`):

| Capture | True spaces | Predicted | Split | Merge | Type correct | Score |
|---|---|---|---|---|---|---|
| c7d28f72c6 (held out) | 6 | 8 | 0 | 0 | 6/6 | **1.00** |
| 1a8384c3f6 | 7 | 8 | 2 | 2 | 6/7 | 0.64 |

**Per-room output** (95 % CI on every number; c7d28f72c6):

| Space | Length x width (m) | CI | Area (m2) | Ceiling (m) |
|---|---|---|---|---|
| Bathroom 1 | 2.50 x 2.21 | +-0.019 | 5.5 | 2.374 +- 0.014 |
| Corridor | 0.99 x 0.91 | +-0.017 | 1.0 | 2.438 +- 0.014 |
| Bathroom 2 | 1.92 x 2.01 | +-0.018 | 4.5 | 2.268 +- 0.014 |
| Stairs | 3.50 x 4.08 | +-0.022 | 14.7 | not seen |
| Room 1 | 2.86 x 4.34 | +-0.020 | 13.0 | not seen (< 50 % coverage) |
| Living / Kitchen | 3.44 x 3.03 | +-0.021 | 12.3 | 3.076 +- 0.014 |
| Room 2 | 3.14 x 3.15 | +-0.021 | 12.0 | 2.959 +- 0.014 |
| Room 3 | 1.82 x 3.55 | +-0.018 | 6.5 | not seen |

Ceilings: a height is reported only where a downward plane covers >= 50 % of the room
(`tools/ceiling_view.py` shows the evidence). The ~2.3-2.4 m values are a continuous dropped false
ceiling over the wet rooms and passage; main rooms 2.96-3.08 m. `1a8384c3f6` and `c00a170fe1` never
look up, so every ceiling there is `null` instead of a guess (an earlier version printed 2.17 m from a
shelf underside - fixed).

**Drift accountability** (`python tools/drift_ablation.py <capture>`; ghosting = median gap between
two sightings of the same wall >= 20 s apart):

| Capture | Ghosting off -> on | Wall crispness off -> on | Decision |
|---|---|---|---|
| c7d28f72c6 | 11.8 -> **9.5 mm** | 12.6 -> 11.0 mm | applied |
| 1a8384c3f6 | 56.8 -> **50.9 mm** | 11.4 -> 10.4 mm | applied |
| c00a170fe1 | 63.3 -> 54.2 mm | 5.8 -> 7.8 mm (worse) | rejected (auto gate) |

Footprint images with correction on/off: `outputs/<capture>/drift_ablation.png`.

**Repeatability** (same flat, two LiDAR captures; `python tools/repeatability.py c7d28f72c6 1a8384c3f6 [--rooms]`):

| Level | Result | Gate |
|---|---|---|
| Structural walls (>= 2.5 m solid), after registration | median 3.1 cm | - |
| Room dimensions, 4 rooms segmented alike in both (8 dims) | median **17.2 cm**, 0/8 within 1 cm or 0.5 % | **fail** |

We are **unrepeatable, not biased**: the walls agree to ~3 cm; room dimensions differ because the
two captures pick different surfaces as a room's boundary (wardrobe/vanity front vs the wall behind).
See the fix loop.

**Openings / ceiling vs laser:** no laser ground truth exists for the sample flat, so the <= 2 cm
opening and <= 1.5 cm ceiling gates cannot be scored there. Reported CIs: openings +-1.6-2 cm,
ceilings +-1.4 cm.

## Video tier (RGB only)

Scale source chosen by measurement (`python tools/depth_scale_eval.py`, 40 level frames, 3 captures,
predicted depth / LiDAR depth):

| Model | Median ratio | 68 % band | Per capture |
|---|---|---|---|
| MapAnything (own metric head) | 0.76 | 0.60-0.91 | - |
| DA3-Metric-Large (true focal) | 0.929 | 0.87-0.99 | 0.91 / 0.93 / 0.94 |
| MoGe-2, free FOV | 0.920 | 0.83-1.02 | 0.89 / 0.92 / 0.94 |
| **MoGe-2, known FOV** | **0.955** | 0.88-1.01 | **0.956 / 0.947 / 0.955** |

MoGe-2 is used, divided by 0.955 (a calibration, not a fit to a test room). Its own focal estimate
on the videos: 47.8 deg (true ~48.6) and 1582 px (true ~1598 px).

Pose source: COLMAP/GLOMAP fragment these indoor videos (largest model 97 of 885 frames even with a
3x more sensitive SIFT) -> MapAnything windows, each conditioned on the already-solved poses of its
overlap frames.

| Capture (RGB only) | Depth vs LiDAR | Trajectory vs ARKit | Plan vs LiDAR plan |
|---|---|---|---|
| c00a170fe1 | 1.02x (p10-p90 0.87-1.11) | scale 5.7 % short, 55 cm median path error | overall 7.88 x 5.82 m vs **7.99 x 6.02 m** (-1.4 %, -3.3 %); Room 4.42 x 1.83 vs 4.45 x 1.87 m; bathroom merged into a neighbour |
| c7d28f72c6 | pending | pending | pending |
| home video (iPhone 17) | - | - | Room2 vs tape: pending |

Failure modes: pose noise (10-50 cm) blurs interior walls, so small rooms merge and the footprint
overfills; fast pans (the home video: 165 s for a whole 3-bed flat) break windows.

## Photo tier

Per-room folders; each room planned from its own photos, rooms joined through doorway photos.

| Set | Photos / folders | Rooms placed by doorway links | Result |
|---|---|---|---|
| c7d28f72c6 cut set | 44 / 7 (14 doorway shots) | 5 of 7 (link residuals 0.7-6 cm) | per-room dims vs LiDAR: pending |
| home photos (iPhone 17) | 46 / 10 | taken before the doorway rule: few links | Room2 vs tape: pending; early run ceiling 2.67 vs 2.718 m (-1.8 %) |

## Damage

| Test | Result |
|---|---|
| BD3 public test split (793 images, 7 classes) | 90.2 % class accuracy; defect vs plain **99.4 %** (0 % plain flagged, 0.75 % defects missed) |
| Clean sample flat c7d28f72c6, 137 m2 of wall/ceiling | 1 region flagged (faint light patch) after excluding floors and gating fixtures |
| Real cracked bedroom photo (`data/damage_tests/`) | both walls' cracks + cornice cracks found; furniture/clothes gated out; one mild spalling (confirmed by owner) |
| Home Room3 (real damp/peeling patch) | pending |

## Timing (cold run, this machine)

| Capture | Tier | Plan | Damage | Total |
|---|---|---|---|---|
| c00a170fe1 | LiDAR | 47 s | ~1 min | ~2 min |
| c7d28f72c6 | LiDAR | 109 s | ~5 min | ~7 min |
| 1a8384c3f6 | LiDAR | 138 s | ~5 min | ~7 min |
| c00a170fe1 (38 s video) | Video | ~12 min | - | ~12 min |
| c7d28f72c6 (3.5 min video) | Video | pending | | |
| 44 photos | Photo | ~5 min | ~2 min | ~7 min |

## Head-to-head vs magicplan / Polycam
Not done at the LiDAR tier: no iPhone Pro was available to scan benchmark rooms with both Stray
Scanner and an incumbent app. Planned substitute: magicplan (non-LiDAR AR mode) vs our video tier vs
tape on two rooms of the home flat - see the table below if present.
