# Capture protocol (Route 2: stock app) — one page

**You need:** an iPhone 15 or newer. *LiDAR tier*: iPhone Pro / Pro Max (12 Pro or later works). *Video and photo tiers*: any iPhone 15+.

## LiDAR tier — Stray Scanner (free, App Store: "Stray Scanner", by Stray Robots)
1. **Install** Stray Scanner. Open it once and allow camera access. In *Settings*, leave **Depth** and **Confidence** ON (the defaults) and set **FPS to 60** (the default).
2. **Before you start:** switch on all the lights, open every internal door fully, and wipe the camera lens. Mirrors and glass are fine: just don't stand still facing one.
3. **Start** in the room by the front door. Hold the phone **upright (portrait), at chest height, screen facing you**. Press record.
4. **Walk slowly.** Take about one step per second along the walls, keeping about 1–2 m from them. In each room:
   - Do one slow full turn so every wall is seen.
   - **Ceiling sweep:** tilt the phone up until the ceiling fills the screen, count *one-two-three*, then tilt it back. Do this once per room. It is required for ceiling height.
   - **Floor sweep:** tilt it down to the floor once.
   - Point at each door and window for a second, so the whole frame is in view.
5. **Go through every doorway**, walking through the middle. Don't walk backwards through doors.
6. **Finish where you started**, in the first room, and look around it again for 5 seconds. This closes the loop for drift correction. Then stop recording.
7. **Length:** about 1 minute per room. Keep one recording **under 6 minutes**. For a bigger property, record one floor per file.

**Avoid:** running or turning quickly; covering the top-back of the phone (that's where the LiDAR is); filming with people walking in front of you; pausing for more than 10 s.

## Video tier — native Camera app
Use **Video, 4K 30 fps, 1× lens, portrait**, and the same walk as above (steps 2–7). Keep the phone level and do the ceiling and floor sweeps.

## Photo tier — native Camera app
Use one album per room, named after the room (e.g. `Kitchen`). In each room take **4–8 photos**: one from each corner looking diagonally across the room, plus one of each doorway taken from the doorway. Keep photos 1×, portrait, with no zoom and no portrait mode. In each doorway photo, part of the next room must be visible: that overlap is what stitches rooms together.

## Hand the files to the pipeline
- **Stray Scanner:** *Library → select the scan → Share → Save to Files*, or AirDrop it to the Mac. You get a folder with `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv` and `imu.csv`.
- **Video and photos:** AirDrop them to the Mac. Put photos in one folder per room.
- **Run:** `python run_capture.py <folder>`. This writes `outputs/<folder>/plan.json` and `plan.png`.
