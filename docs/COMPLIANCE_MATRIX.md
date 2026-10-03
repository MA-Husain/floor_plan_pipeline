# Compliance matrix - requirement -> file path -> artifact -> status

Status: **Done** (built and measured) - **Partial** (built, gate not met or evidence incomplete) - **Missing**.

## Part 1 - capture route and tiers
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Route 2: one-page stock-capture protocol | `docs/CAPTURE_PROTOCOL.md` | protocol for LiDAR (Stray Scanner), video and photo | Done |
| Three input tiers, same output contract | `run_capture.py`, `brynz/lidar.py`, `brynz/rgb.py`, `brynz/photo.py` | tier auto-detected; one `plan.json` schema | Done |
| Photo tier: per-room folders -> stitched plan | `brynz/rgb.py` (`photo_stitch`), `brynz/photo.py` | rooms joined through doorway photos; unlinked rooms flagged | Partial (stitching needs doorway photos) |
| Video tier from any iPhone 15+ | `brynz/rgb.py` (`video_capture`), `brynz/mono.py` | MoGe-2 metric depth + MapAnything poses | Partial (accuracy below LiDAR; see benchmark) |
| LiDAR tier | `brynz/capture.py`, `brynz/lidar.py` | Stray Scanner -> plan | Done |
| Device matrix | `docs/DEVICE_MATRIX.md` | tier x device x accuracy x run time | Done |

## Part 2 - output contract and gates
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Dimensioned per-room plan (walls, ceiling, area, openings) | `brynz/lidar.py` | `rooms[].dimensions`, `ceiling_height`, `area_m2`, `openings[]` | Done |
| Stitched multi-room plan, adjacency | `brynz/cells.py`, `brynz/spaces.py`, `brynz/render.py` | `adjacency`, `plan.png` | Done (LiDAR, video); Partial (photo) |
| Per-surface damage regions, class + metric extent | `brynz/damage.py`, `models/defect_probe.npz` | `damage.regions[]` (surface, room, m2 +- CI) | Done |
| Concealed-damage flags with rule fired | `brynz/damage.py` (`concealed_flags`) | `concealed_damage_flags[]` (R1-R6) | Done |
| Scope line items keyed to surfaces | `brynz/damage.py` (`scope_items`) | `scope[]` | Done |
| CI on every measurement | `brynz/lidar.py` (error model), `brynz/rgb.py` (tier error models) | `ci95` fields; `error_model` | Done |
| One command per capture, JSON, rendered plan | `run_capture.py`, `brynz/render.py` | `outputs/<name>/plan.json`, `plan.png` | Done |
| Openings <= 2 cm on >= 85 % | `benchmark/` | needs laser GT of openings | Missing (no opening GT) |
| Ceiling <= 1.5 cm per room; spread <= 1 cm | `brynz/lidar.py` (`room_ceiling`), `tools/ceiling_view.py` | ceiling reported only where seen; tape 2.718 m for home Room2 | Partial |
| Repeatability <= 1 cm / 0.5 % per wall | `tools/repeatability.py` | `docs/fixloop/*_rooms.txt` | Partial (17.2 cm median: fails) |
| Drift accountability, ablation on/off | `brynz/drift.py`, `tools/drift_ablation.py` | `outputs/<capture>/drift_ablation.png/.json` | Done |
| Photo-tier whole-property stitch, footprint +-8 % | `brynz/photo.py`, `tools/make_photo_set.py` | photo sets cut from LiDAR captures, LiDAR as truth | Partial |
| Photo +-8 % / video +-3 % wall lengths, calibrated | `tools/rgb_vs_lidar.py`, `benchmark/eval_tape.py` | see `docs/BENCHMARK.md` | Partial |

## Constraints
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Offline, any pretrained model with disclosure | `README.md` (models table), `run_capture.py` (HF offline) | no network at run time | Done |
| Weights fetched by script | `scripts/fetch_weights.py` | MapAnything, MoGe-2, CLIP, YOLO-World, code clones | Done |
| Mirrors, glass, wet-look surfaces, low light | `docs/CAPTURE_PROTOCOL.md` (capture rules), `docs/TECHNICAL_REPORT.md` s8, `docs/DEVICE_MATRIX.md` | protocol instructions + stated failure modes | Partial (not handled in code beyond LiDAR confidence filtering) |

## Part 3 - head-to-head
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| LiDAR tier vs Polycam / magicplan on 2 rooms | `docs/BENCHMARK.md` (head-to-head), `benchmark/gt/home.json`, `data/rooms_data/Room1-Room2.pdf` | magicplan (iPhone 17, non-LiDAR AR) vs our video and photo tiers vs tape, 2 rooms | Partial (no Pro device: compared at the RGB tiers, not LiDAR) |

## Part 4 - fix loop
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Declaration (gate, root cause, fix, prediction) | `docs/fixloop/DECLARATION.md` | committed before the fix (tag `fixloop-before`) | Done |
| Before / after runs, regenerable, readable diff | `docs/fixloop/before_rooms.txt`, `after_rooms.txt`, `git diff fixloop-before..` | 20.3 -> 17.2 cm (prediction 4 cm: missed; post-mortem in report) | Done (gate not passed) |

## Part 5 - process
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Commit history | `git log` | incremental commits | Done |

## Deliverables
| Deliverable | File path | Status |
|---|---|---|
| Compliance matrix | `docs/COMPLIANCE_MATRIX.md` | Done |
| Capture route + device matrix | `docs/CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` | Done |
| README, < 15 min to running, one command | `README.md`, `requirements.txt`, `scripts/fetch_weights.py` | Done |
| Reproduction bundle (cached model outputs replay; live path runs) | `outputs/<name>/cache/` (shared via the data link: too large for git), `--live`, `results/` snapshot | Done |
| Benchmark report | `docs/BENCHMARK.md` | see file |
| Fix loop bundle | `docs/fixloop/` | Done |
| Technical report <= 6 pages | `docs/TECHNICAL_REPORT.md` | see file |
| Raw benchmark data | sample captures `c00a170fe1`, `1a8384c3f6`, `c7d28f72c6`; own iPhone 17 data `data/home/`, `data/rooms_data/` (video, photos, magicplan export); `benchmark/gt/` | Partial (tape GT for home Room1 + Room2 only; no laser) |
