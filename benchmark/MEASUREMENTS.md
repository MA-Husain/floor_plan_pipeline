# Ground-truth measurements and app exports

## Own home (iPhone 17, non-Pro) - tape measure by the owner
Inches converted at 0.0254 m/in. Machine-readable: `gt/home.json`.

| Room | Ceiling height | Length | Width | Note |
|---|---|---|---|---|
| Room1 | 107 in = 2.718 m | ~132 in = 3.35 m wall to wall (158 in = 4.01 m including the wardrobe/window alcove) | 135 in = 3.43 m | 132 in from memory (+-5 cm) |
| Room2 | 107 in = 2.718 m | 146 in = 3.708 m | 120 in = 3.048 m | |

Visible damage: Room3, a damp / peeling patch near the skirting (video ~122 s, photos IMG_9122-9128).

## Incumbent app export - magicplan (Sensopia), free tier, iPhone 17 (non-LiDAR AR mode)
File: `gt/magicplan_Room1-Room2.pdf` (top plan = Room1, bottom plan = Room2).

| Room | magicplan size | Openings |
|---|---|---|
| Room1 | 2.90 x 3.29 m (8.98 m2) | door 0.88 m, opening 0.73 m, windows 1.68 m and 0.69 m |
| Room2 | 3.96 x 3.05 m (11.42 m2) | door 0.83 m, window 2.09 m |

## Sample captures (provided)
`c00a170fe1`, `1a8384c3f6`, `c7d28f72c6` (Stray Scanner, iPhone Pro). No tape / laser measurements were
provided; the LiDAR plan is the reference for the video and photo tiers run on the same walks, and
`gt/c7d28f72c6.json`, `gt/1a8384c3f6.json` hold frame-level room labels made by hand from the videos.

## Raw recordings
Too large for git: home video (`data/home/video/IMG_9085.mov`), home photos (`data/home/photos/<room>/`),
Room1/Room2 videos and photos (`data/rooms_data/room1|room2/`) - shared via the link in the submission email.
