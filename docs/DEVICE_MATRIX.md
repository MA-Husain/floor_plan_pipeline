# Device matrix - which tier runs on which iPhone, and what it honestly delivers

| Tier | Capture app | iPhone 15 / 16 / 17 (non-Pro) | iPhone 12 Pro - 17 Pro (LiDAR) | Inputs the pipeline uses |
|---|---|---|---|---|
| **LiDAR** | Stray Scanner (free) | not available (no LiDAR) | yes | LiDAR depth 256x192 + confidence, ARKit poses + intrinsics, RGB video |
| **Video** | native Camera (Video, 1x, 30 fps) | yes | yes | RGB frames only |
| **Photo** | native Camera (Photo, 1x) | yes | yes | RGB stills + EXIF focal length only |

Processing machine for all numbers below: Apple M5, 16 GB, macOS, PyTorch MPS. No network at run time.

## Accuracy each tier delivers (measured; how in brackets)

| Quantity | LiDAR | Video | Photo |
|---|---|---|---|
| Metric scale | sensor (no model) | MoGe-2 depth, calibrated: 1.02x LiDAR on the single-room walk; per-frame spread +-7 %, averaged over hundreds of frames [tools/depth_scale_eval.py, tools/rgb_vs_lidar.py] | MoGe-2 with EXIF focal; per-photo +-7 %, averaged over the room's photos |
| Camera poses | ARKit VIO, 17 cm end drift over 54 m; plane-anchored drift correction on top | MapAnything (pose-conditioned windows): ~10-50 cm path error vs ARKit [tools/rgb_vs_lidar.py] | MapAnything per room; rooms joined only through doorway photos |
| Wall position, single capture | wall crispness 10-12 mm | blurred by pose noise; interior walls can be lost | per room only |
| Room length / width | 95 % CI typically +-2 cm | overall extent within 1.4-3.3 % of LiDAR on the single-room walk; CI +-0.2-0.4 m | see benchmark report; CI widened by 12 % scale term |
| Ceiling height | +-1.4 cm CI, only where the ceiling plane covers >= 50 % of the room | as LiDAR rule, scale term 3 % | as LiDAR rule, scale term 12 % |
| Openings | width CI ~+-2 cm | +-3 % of width | per room |
| Repeatability (same rooms, two captures) | 17.2 cm median per room dimension (gate 1 cm: **fails**, see fix loop) | not measured (one capture) | not measured |

## Error model per tier (95 % intervals are built from these 1-sigma terms; `plan.json` -> `error_model`)

| Term | LiDAR | Video | Photo |
|---|---|---|---|
| face systematic (m) | 0.006 | 0.02 | 0.03 |
| drift per metre of separation | 0.002 | 0.01 | 0.015 |
| relative scale | 0 | 0.03 | 0.12 |

## Run time (cold, no cached model outputs)

| Tier | Typical capture | Plan | + damage | Total |
|---|---|---|---|---|
| LiDAR | 3 min walk, 7-8 rooms | ~2 min | ~5 min | ~7-8 min |
| Video | 40 s single room | ~12 min | ~2 min | ~14 min |
| Video | 3 min walk, 900+ keyframes | ~60 min | ~5 min | see benchmark report |
| Photo | ~45 photos | ~5 min | ~2 min | ~7 min |

## Known limits by device / condition
- Non-Pro iPhones cannot run the LiDAR tier; Pro phones can run all three.
- Video: fast pans (a full turn in under ~4 s) break pose estimation; the protocol asks for ~8 s per turn.
  Feature-based SfM (COLMAP/GLOMAP) was tried and fragments these indoor videos (<= 97 frames per model).
- Photo: rooms are joined only through doorway photos that show the next room; without them a room is
  still measured but laid out separately and flagged `position_known: false`.
- Mirrors and glass: LiDAR returns through glass are dropped by the confidence filter; mirrors create
  phantom rooms behind them in RGB tiers (known failure mode, not yet handled).
- Low light: blur lowers keyframe sharpness; the protocol asks for all lights on.
