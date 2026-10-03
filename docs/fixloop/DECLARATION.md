# Fix declaration (written before the fix; committed separately)

## 1. Worst-performing gate
**Repeatability** - two captures of the same rooms at the same tier (LiDAR: `c7d28f72c6` and
`1a8384c3f6`, the same flat) must agree within 1 cm or 0.5 % per wall.

Before: **0 / 8** room dimensions pass; **median difference 20.3 cm** (4 rooms that are segmented the
same way in both captures; 2 more are split/merged and excluded). Regenerate:
`python tools/repeatability.py c7d28f72c6 1a8384c3f6 --rooms` -> `docs/fixloop/before_rooms.txt`.

## 2. Root-cause hypothesis and evidence
The walls themselves are repeatable; the **definition of a room dimension** is not.
`room_dimensions()` reports, per axis, the distance between the two *extreme* polygon edges of the
segmented space. Anything that perturbs the outline at its extremes - an alcove or door reveal
included in one capture and not the other, a wardrobe front taken as the boundary, a sliver of the
neighbouring corridor - moves the dimension by tens of centimetres, even though the room's main walls
did not move.

Evidence:
- Face level, independent of segmentation: after registering the two captures, the structural walls
  (>= 2.5 m of solid wall) agree to a median of **3.1 cm** (`docs/fixloop/before_faces.txt`).
- Room level the same walls give a median of **20.3 cm**, a 6.5x gap that only the dimension
  definition and outline can create. Examples: `study` long side 4.98 m vs 6.16 m (the outline in
  `1a8384c3f6` includes an L-shaped extension); `bath_b` short side 2.01 vs 2.26 m.
- A tape or laser measurement - what the walk-in test uses - is taken between the two main
  opposing walls of a room, not between the farthest corners of its outline.

## 3. The fix and the predicted number
Measure each room dimension between its **principal opposing walls**: for each axis, on each side of
the room take the wall line carrying the most *measured* wall length along the room's outline (not the
extreme edge), and report the distance between the two. Keep the old extents as `extent_u/extent_v`
for the footprint. No thresholds tuned on the benchmark.

Prediction: median per-room difference **20.3 cm -> about 4 cm**; pass rate **0/8 -> about 2/8**.
The 1 cm gate itself is not expected to pass: the face-level floor is ~3 cm, set by the two
captures' independent ARKit drift and by single-capture wall-plane noise (crispness 10-12 mm).
