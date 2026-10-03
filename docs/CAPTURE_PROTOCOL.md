# Capture protocol (Route 2: stock apps) - one page

**Pick the tier by phone.** iPhone **Pro / Pro Max** (12 Pro or later): LiDAR, video or photo tier. **Any other iPhone 15 or newer**: video or photo tier.
All three give the same kind of plan; LiDAR is the most accurate, photos the least (`DEVICE_MATRIX.md`).

**Before any capture (all tiers):** switch on every light and open the curtains; open every internal door fully; wipe the lens;
ask people and pets to stay out of the way. **Mirrors, glass doors, shiny or wet floors:** fine to film, but never stand still
facing one - keep moving past it. **Dark rooms:** switch the lights on; if a room has no light, use the phone torch of a second phone.

## A. LiDAR tier - Stray Scanner (free; App Store: "Stray Scanner" by Stray Robots)
1. Install it, open it once, allow camera access. Leave its settings at the defaults (Depth and Confidence ON).
2. Stand in the room by the front door. Hold the phone **upright (portrait), at chest height, screen facing you**. Tap the record button.
3. In every room: walk slowly along the walls, about 1-2 m away from them (one step per second); do **one slow full turn** (about 8 s);
   then the **ceiling sweep** (tilt the phone up until the ceiling fills the screen, count *one-two-three*, tilt back) and one **floor sweep**.
   Point the phone at each door and window for one second so the whole frame is in view.
4. Walk **through the middle of every doorway**, facing forwards.
5. **Finish where you started** and look around that first room again for 5 seconds. Tap stop.
6. About 1 minute per room; keep one recording **under 6 minutes** (one recording per floor).

## B. Video tier - native Camera app (any iPhone 15+)
1. Camera app -> **Video**. Use the **1x** lens (not 0.5x), leave resolution at the default, hold the phone **upright (portrait)**.
2. Do the same walk as A, steps 2-6 (ceiling sweep and floor sweep in every room, finish where you started).
3. **Turn slowly: one full turn takes about 8 seconds.** Do not zoom, do not switch lenses, do not run.

## C. Photo tier - native Camera app (any iPhone 15+)
1. Camera app -> **Photo**, **1x** lens, phone **upright (portrait)**. No zoom, no Portrait mode, no Pano.
2. For every room, take **4 to 8 photos** while standing **inside that room**:
   - one from **each corner**, looking diagonally across the room (some floor and some ceiling in view);
   - **one per doorway**: stand in the doorway, looking **into the next room** (part of the next room must be visible). This photo
     belongs to the room you are leaving - it is what joins the rooms into one plan.
   - If corners + doorways exceed 8, drop corner photos (keep all doorway photos). Small rooms (WC, closet): 2-3 photos from the doorway are enough.
3. Photograph one room completely before moving to the next.

## Hand the files to the pipeline
- **LiDAR:** Stray Scanner -> Library -> tap the scan -> Share -> AirDrop to the Mac (or Save to Files). You get one folder.
- **Video:** Photos app -> the video -> Share -> AirDrop to the Mac. You get one `.MOV` file.
- **Photos:** on the Mac make one folder for the property (`my_flat`) and inside it **one folder per room** named after the room
  (`Kitchen`, `Bedroom1`, `Bathroom`, ...). AirDrop each room's photos and drag them into that room's folder.
- **Run (one command):** `python run_capture.py <the Stray folder | the .MOV | my_flat>` -> `outputs/<name>/plan.json` and `plan.png`.

**Avoid:** running or fast turns; walking backwards through doors; people walking in front of the camera; pausing for more than 10 s;
covering the top-back of the phone (LiDAR sensor); filming straight at a mirror or window for more than a moment.
