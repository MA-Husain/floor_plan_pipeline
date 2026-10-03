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
| home video (iPhone 17, fast walk: whole 3-bed flat in 165 s) | - | - | Room2 6.05 x 4.84 m vs tape 3.71 x 3.05 m: **+60 %, outside the CI** - a capture far faster than the protocol (8 s per turn) breaks the pose windows; the plan does not flag it (known failure: no capture-quality check) |

Failure modes: pose noise (10-50 cm) blurs interior walls, so small rooms merge and the footprint
overfills; fast pans (the home video: 165 s for a whole 3-bed flat) break windows.

## Photo tier

Per-room folders; each room planned from its own photos, rooms joined through doorway photos.

| Set | Photos / folders | Rooms placed by doorway links | Result |
|---|---|---|---|
| c7d28f72c6 cut set | 44 / 7 (14 doorway shots) | 5 of 7 (link residuals 0.7-6 cm) | see per-room table below |
| home photos (iPhone 17) | 46 / 10 | photos taken before the doorway rule: 2 rooms linked, 8 unplaced (flagged) | Room2 vs tape below; an earlier version measured its ceiling 2.67 vs 2.718 m (-1.8 %) |

Per-room dimensions, photo tier vs reference (`python run_capture.py data/photo_sets/c7d28f72c6`):

| Room (c7 cut set) | Photo tier (m) | 95 % CI | LiDAR (m) |
|---|---|---|---|
| Bathroom 1 | 1.42 x 2.48 | +-0.38 | 2.50 x 2.21 |
| Bathroom 2 | 2.26 x 3.06 | +-0.56 | 1.92 x 2.01 |
| Room 1 | 4.30 x 4.14 | +-1.03 | 2.86 x 4.34 |
| Stairs | 2.14 x 6.36 | +-0.53 | 3.50 x 4.08 |
| Room 2 | 0.37 x 8.20 (failed) | +-0.12 | 3.14 x 3.15 |
| Corridor, Living/Kitchen | no room found | - | 0.99 x 0.91, 3.44 x 3.03 |

Home Room2 vs tape (`python benchmark/eval_tape.py outputs/home_photos_v2/plan.json benchmark/gt/home.json`):
length 4.83 +- 1.16 m vs 3.708 (+30 %, inside CI), width 3.18 +- 0.76 vs 3.048 (+4.3 %, inside CI).

**Verdict: the photo tier runs end to end and its intervals are honest (the truth falls inside them),
but it does not meet the +-8 % gate: per-room errors are 4-50 %, and on sparse stills the room
outline is often bounded by what the photos happened to see.**

## Damage

| Test | Result |
|---|---|
| BD3 public test split (793 images, 7 classes) | 90.2 % class accuracy; defect vs plain **99.4 %** (0 % plain flagged, 0.75 % defects missed) |
| Clean sample flat c7d28f72c6, 137 m2 of wall/ceiling | 1 region flagged (faint light patch) after excluding floors and gating fixtures |
| Real cracked bedroom photo (`data/damage_tests/`) | both walls' cracks + cornice cracks found; furniture/clothes gated out; one mild spalling (confirmed by owner) |
| Home Room3 (real damp/peeling patch near the skirting, ~5 % of the photo) | **missed** (`results/damage_photo/IMG_9125.jpg`): 4 of 703 tiles call it a stain, below the region vote; small defects are a known failure mode |

## Timing (cold run, this machine)

| Capture | Tier | Plan | Damage | Total |
|---|---|---|---|---|
| c00a170fe1 | LiDAR | 47 s | ~1 min | ~2 min |
| c7d28f72c6 | LiDAR | 109 s | ~5 min | ~7 min |
| 1a8384c3f6 | LiDAR | 138 s | ~5 min | ~7 min |
| c00a170fe1 (38 s video) | Video | ~12 min | - | ~12 min |
| c7d28f72c6 (3.5 min video) | Video | pending | | |
| 44 photos | Photo | ~5 min | ~2 min | ~7 min |

## Head-to-head vs magicplan
No iPhone Pro was available, so the LiDAR-tier comparison could not be run. Substitute on two rooms of
the home flat, all on an iPhone 17 (non-Pro): **magicplan** (Sensopia, free tier, AR mode; export
`data/rooms_data/Room1-Room2.pdf`) vs **our video and photo tiers** (`data/rooms_data/room1|room2`) vs a
tape measure (`benchmark/gt/home.json`; Room1 length ~132 in is from memory, +-5 cm).

Dimensions are compared longer-to-longer and shorter-to-shorter (axis naming differs between tools).

| Room | Dimension | Tape (m) | magicplan (m) | magicplan error | Ours, photo tier (m) | Our error | Better |
|---|---|---|---|---|---|---|---|
| Room1 | longer | 3.43 | 3.29 | -4.1 % | 3.75 +- 0.89 | +9.4 % | magicplan |
| Room1 | shorter | 3.35 | 2.90 | -13.5 % | 3.34 +- 0.89 | **-0.4 %** | **ours** |
| Room2 | longer | 3.71 | 3.96 | +6.8 % | 4.07 +- 0.98 | +9.8 % | magicplan |
| Room2 | shorter | 3.05 | 3.05 | +0.1 % | 2.97 +- 0.98 | -2.6 % | magicplan |

Ours beat or tied on **1 of 4** shared dimensions (gate: 70 %) - **fail**. Both photo-tier rooms were
stitched through a doorway photo (link residual 2.7 cm). magicplan uses ARKit tracking live on the phone;
our photo tier has 9 stills per room and no poses. Video-tier rows: see below if the runs completed.
