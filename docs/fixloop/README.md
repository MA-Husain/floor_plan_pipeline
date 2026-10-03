# Fix loop bundle

| File | What |
|---|---|
| `DECLARATION.md` | gate, failing number, root-cause hypothesis + evidence, fix, predicted number - committed before the fix |
| `before_rooms.txt`, `before_faces.txt` | the before run (code at tag `fixloop-before`) |
| `after_rooms.txt` | the after run |

Result: **20.3 cm -> 17.2 cm** median per-room dimension difference between two captures of the same
flat; 0/8 -> 0/8 within the 1 cm gate. Prediction was ~4 cm: missed. Post-mortem: technical report,
section 6 (the deeper root cause is furniture faces accepted as room boundaries).

## Regenerate

```bash
# before
git checkout fixloop-before
python run_capture.py c7d28f72c6 --no-damage && python run_capture.py 1a8384c3f6 --no-damage
python tools/repeatability.py c7d28f72c6 1a8384c3f6 --rooms     # -> median 20.3 cm, 0/8

# after (the fix commit; main gives the same LiDAR numbers)
git checkout 9303deb
python run_capture.py c7d28f72c6 --no-damage && python run_capture.py 1a8384c3f6 --no-damage
python tools/repeatability.py c7d28f72c6 1a8384c3f6 --rooms     # -> median 17.2 cm, 0/8
```

## Readable diff

```bash
git show 9303deb -- brynx/lidar.py     # the fix commit: room_dimensions() extremes -> principal measured walls
```
