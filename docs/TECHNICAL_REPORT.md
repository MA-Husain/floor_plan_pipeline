# Technical report - phone capture to dimensioned, damage-annotated floor plan

## 1. What the system does
`python run_capture.py <capture>` takes a Stray Scanner LiDAR folder, an iPhone walkthrough video,
or a folder of per-room photo folders, and writes one `plan.json` (schema `brynz.plan/1.0`) plus a
rendered architectural plan: rooms with length/width, area, ceiling height and type; doors, open
passages and windows with widths; adjacency; per-surface damage regions with metric extent;
concealed-damage flags with the rule that fired; scope line items; a 95 % interval on every number.
Everything runs offline from weights fetched once (`scripts/fetch_weights.py`).

## 2. Architecture
```
capture ──► tier adapter ──────────────► common capture interface ──► plan core ──► damage ──► JSON + render
            LiDAR : Stray depth + ARKit poses            (frames: depth, confidence,   │
            video : MoGe-2 depth + MapAnything poses       K, cam-to-world pose)       ├─ drift correction (pose graph)
            photo : MoGe-2 depth + per-room MapAnything,                               ├─ floor/ceiling, Manhattan frame
                    rooms joined via doorway photos                                    ├─ wall faces, door-gap barrier
                                                                                       ├─ cell complex -> spaces
                                                                                       ├─ typing (structure, fixtures)
                                                                                       └─ dimensions + CIs
```
The decisive design choice is the **common capture interface**: every tier produces per-frame depth,
confidence, intrinsics and a camera pose. The plan core, drift handling, damage and rendering are the
same code for all tiers; only the error model differs. A bug fixed in the core is fixed for all tiers.

**Plan core.** Depth is back-projected with per-pixel normals from the organised depth image
(discontinuities rejected) and voxel-reduced keeping the closest-range point. Floor and ceiling come
from the normal-filtered height histogram; the Manhattan yaw from a 4-fold histogram of wall-normal
angles. Wall faces are extracted as peaks of oriented vertical points with solid intervals by height
coverage (a face must cover >= 55 % of the observed height band - furniture rarely does). Door-sized
gaps (0.55-1.30 m, residential norms) are bridged into a barrier so doors separate spaces while open-plan
passages do not. Faces cut the floor into a Manhattan cell complex; cells are inside if seen as floor,
carved free by camera rays, or walked; cells are grouped by watershed over the barrier, then necks
(<= 2.0 m between two >= 3 m2 sides) and cores are split, slivers merged and never-observed voids removed.
Every threshold lives in `brynz/priors.py` with its building-norm justification; none is fitted to the
benchmark, and `c7d28f72c6` was held out throughout.

**Typing.** Structure first: narrow and elongated = corridor; small single-entry = closet; many
connections and no fixtures = hall; floor dropping below level = stairs. Fixtures (YOLO-World,
detections lifted to 3-D and clustered into instances seen in >= 2 views) decide the rest, but only
*decisive* fixtures name a room (toilet/shower -> bathroom, stove/fridge -> kitchen, bed -> bedroom,
sofa -> living). Desks, TVs and tables never do: when in doubt the space is called "Room".

**Dimensions.** A room's length/width is the distance between its *principal opposing walls* (the
wall line carrying the most measured length on each side), as a tape would be laid - not between the
outline's extreme corners (fix loop, section 6). A dimension whose end is not a measured wall is
flagged. Ceiling height is reported only when a downward plane covers >= 50 % of the room's area;
otherwise `null` - never a shelf underside or a guess.

## 3. Tier design and device matrix
Full matrix: `docs/DEVICE_MATRIX.md`.

**LiDAR** (iPhone Pro, Stray Scanner). Depth 256x192 with confidence 2 only, ARKit poses. The old
pipeline's ICP "drift corrector" corrupted these poses (frames after the last closure were left
uncorrected); raw ARKit drifts only 17 cm over 54 m, so drift handling is now a gated correction on
top of ARKit, never a replacement.

**Video** (any iPhone 15+, RGB only). Measured, not assumed, at each step:
- *Scale*: MoGe-2 metric depth with a known field of view reads 0.955x LiDAR on all three captures
  (0.947-0.956), MapAnything's own metric head 0.76x +-20 %; MoGe-2 is used and divided by 0.955.
  MoGe-2 also estimates the video's focal length (47.8 deg vs true ~48.6 deg).
- *Poses*: COLMAP/GLOMAP fragment these texture-poor handheld videos (largest model 97/885 frames).
  MapAnything windows of 24 keyframes are used; each new window is fed its 8 overlap frames *with
  their already-solved poses* plus MoGe depth and the fixed focal, so windows are reconstructed in a
  consistent frame and joined rigidly (independent windows joined afterwards failed on ~20 % of links).
- *Keyframes* are motion-adaptive: a new keyframe once the view has shifted 12 % of the image (phase
  correlation on a thumbnail) or 0.5 s passed - dense on fast pans, sparse when slow.

**Photo** (any iPhone 15+, 2-8 stills per room). Each room folder is reconstructed on its own with
MoGe-2 depth and the EXIF focal length, then planned on its own; its room is the segmented space
containing the folder's camera positions (photos see through doorways, so whole-capture
segmentation cannot tell whose space is whose). Rooms are joined through *guest views*: a doorway
photo from room A that shares SIFT inliers with room B is reconstructed in both rooms' runs; the same
pixels in two frames give dense 3-D correspondences and a rigid link. Rooms are placed along the
maximum-confidence spanning tree and snapped to the building's Manhattan axes. A room with no
doorway link is still measured, laid out beside the plan and flagged `position_known: false`.

## 4. Drift handling
Plane-anchored pose graph (`brynz/drift.py`): ~1 s segments of wall points are ICP'd against the map
from temporally distant segments (>= 20 s apart; implausible corrections > 4 deg or 25 cm rejected);
per-keyframe corrections (x, z, yaw) are solved by tridiagonal least squares (odometry smoothness +
weighted anchors) and interpolated to all frames. The correction is applied only if *revisit ghosting*
(median gap between two sightings of the same wall >= 20 s apart) drops >= 5 % without walls getting
blurrier. Ablation: c7 11.8 -> 9.5 mm, 1a8 56.8 -> 50.9 mm (applied); single room rejected
(`tools/drift_ablation.py`, footprints on/off in `outputs/<capture>/drift_ablation.png`).

## 5. Error budget and calibration
Each tier carries a 1-sigma error model (`plan.json -> error_model`); every interval is
`1.96 * sqrt(face_a^2 + face_b^2 + (drift_per_m * L)^2 + (scale_rel * L)^2)`, with face terms from
the measured face scatter (x3 for correlated points) floored by the tier's systematic term.

| Term (1 sigma) | LiDAR | Video | Photo | Source |
|---|---|---|---|---|
| face systematic | 6 mm | 20 mm | 30 mm | LiDAR range bias 0.5-1 cm; RGB depth noise |
| drift per metre | 0.2 % | 1 % | 1.5 % | ARKit residual; MapAnything pose noise |
| relative scale | 0 | 3 % | 12 % | MoGe-2 calibration spread; per-photo +-7 % on few photos |

Calibration checks: (1) single-room video plan overall 7.88 x 5.82 m vs LiDAR 7.99 x 6.02 m - errors
1.4 % and 3.3 %, inside the video intervals; (2) LiDAR room CIs (+-2 cm) are too tight for
repeatability (17 cm between captures): the systematic error is in *which surface is the wall*, not in
measurement noise - a known under-coverage of our LiDAR intervals, stated rather than hidden;
(3) home Room2 tape vs video/photo: see `docs/BENCHMARK.md`.

## 6. The fix loop
**Gate**: repeatability (worst). Two LiDAR captures of the same flat: 0/8 room dimensions within
1 cm/0.5 %, median 20.3 cm. **Hypothesis**: the walls are repeatable (structural faces agree to 3.1 cm
after registration) but the *dimension definition* (polygon extremes) is not: alcoves, reveals and
segmentation slivers move the extremes. **Fix**: principal-wall dimensions (+ flag for unmeasured
ends). **Prediction**: ~4 cm, 2/8 passing. **Result**: 17.2 cm, 0/8 - correct direction, far short of
the prediction. **Post-mortem**: after the fix all 8 dimensions were bounded by measured walls in both
captures, yet differed by 8-58 cm; an overlay of both registered plans shows the two captures bound the
same room by *different measured surfaces* (a wardrobe/vanity front in one, the wall behind in the
other) and see different extents of partially covered rooms. The root cause is surface selection
(furniture faces accepted as walls), one level deeper than declared. Next fix: a face counts as a room
boundary only if it reaches the ceiling band or continues behind furniture in the free-space carving.
Before/after: `docs/fixloop/`, tag `fixloop-before`, `git diff fixloop-before -- brynz/lidar.py`.

## 7. Damage
Water stains, mould, peeling and cracks are textures, which open-vocabulary detectors handled poorly
(Grounding DINO missed most sample defects). Every measured wall and ceiling is cut into 0.4 m cells
in its own metric coordinates; each cell is cropped from its best 2-3 views (close, frontal, unoccluded
per depth) and classified by a logistic probe on CLIP ViT-L/14 features trained on BD3 (CC-BY-4.0):
90.2 % 7-class, 99.4 % defect-vs-plain on its held-out split. A zero-shot "bare surface" gate drops
sockets, cables and pictures; floors are excluded (tiles read as stains). Adjacent positive cells form
a region with metric area +- CI (boundary cells +-50 %). Rules R1-R6 (ceiling moisture, rising damp,
wet-room wall, condensation at window, structural crack, spalling) raise concealed-damage flags; scope
items are keyed to the surface. On the clean sample flat: 1 false region over 137 m2.

## 8. Known failure modes
- Repeatability of room dimensions (furniture faces as boundaries) - LiDAR intervals under-cover it.
- Video: pose noise blurs interior walls (small rooms merge, footprint overfills); fast pans break it.
- Photo: rooms join only through doorway photos; per-room scale +-7-12 %.
- Mirrors create phantom space in RGB tiers; glass drops out of LiDAR (confidence filter).
- Low light and blur lower keyframe quality; the protocol asks for lights on and slow turns.
- Openings and ceilings have no laser ground truth on the sample flat; head-to-head vs an incumbent
  app needs a Pro iPhone we did not have.
- Run time on a 16 GB Mac: LiDAR ~7 min, photos ~7 min, video ~10-15 min per minute of video.
