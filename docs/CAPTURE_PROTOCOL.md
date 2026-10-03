# Capture protocol (Route 2: stock apps) - one page

**Pick the tier by phone.** iPhone **Pro / Pro Max** (12 Pro or later): LiDAR tier. **Any iPhone 15 or newer**: video tier, or photo tier.
All three give the same plan; LiDAR is the most accurate, photos the least (see `DEVICE_MATRIX.md`).

**Before any capture (all tiers):** switch on every light and open curtains; open every internal door fully;
wipe the lens; move people and pets out of the way. Mirrors and glass are fine - never stand still facing one.

## A. LiDAR tier - Stray Scanner (free, App Store "Stray Scanner" by Stray Robots)
1. Install, open once, allow camera. Settings: leave **Depth** and **Confidence** ON (defaults).
2. Start in the room by the front door. Phone **upright (portrait), chest height, screen facing you**. Tap record.
3. In each room: walk slowly along the walls about 1-2 m from them (one step per second), do **one slow full turn**,
   then the **ceiling sweep** (tilt up until the ceiling fills the screen, count *one-two-three*, tilt back) and one **floor sweep**.
   Point at every door and window for one second so the whole frame is in view.
4. Walk **through the middle of every doorway**, facing forwards.
5. **Finish where you started** and look around that first room again for 5 s (closes the loop). Stop.
6. About 1 minute per room, **under 6 minutes per recording**; one recording per floor.

## B. Video tier - native Camera app (any iPhone 15+)
1. Camera -> **Video**, **1x** lens (not 0.5x), **30 fps**, portrait. Same walk as A, steps 2-6.
2. **Turn slowly - a full turn should take about 8 seconds.** Fast pans blur the frames and break the reconstruction.
3. Keep the phone level; do the ceiling and floor sweep in every room; do not zoom; do not switch lenses.

## C. Photo tier - native Camera app (any iPhone 15+)
1. Camera -> **Photo**, **1x**, portrait, no zoom, no Portrait mode, no Live-photo effects. Keep HEIC or JPEG - both work.
2. In each room take **4-8 photos**: one from each corner looking diagonally across the room (walls, floor and some
   ceiling in view), plus **one photo through each doorway, standing in the doorway, showing part of the next room** -
   that overlap is what stitches the rooms into one plan.
3. Keep each room's photos together: make one album per room named after the room (`Kitchen`, `Bedroom1`, ...).

## Hand the files to the pipeline
- **Stray Scanner:** Library -> select the scan -> Share -> Save to Files (or AirDrop). You get a folder with `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`.
- **Video:** AirDrop the `.MOV` to the Mac.
- **Photos:** AirDrop each album into its own folder: `my_flat/Kitchen/*.HEIC`, `my_flat/Bedroom1/*.HEIC`, ...
- **Run:** `python run_capture.py <the folder or the .MOV>` -> `outputs/<name>/plan.json` and `plan.png`.

**Avoid:** running, fast turns, walking backwards through doors, people in front of the camera, pausing over 10 s,
covering the top-back of the phone (LiDAR), very dark rooms (switch lights on), filming straight into a window or mirror.
