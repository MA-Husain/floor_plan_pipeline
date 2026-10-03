# Compliance matrix - requirement -> file path -> artifact -> status

Status: **Done** (built and measured) - **Partial** (built, gate not met or evidence incomplete) - **Missing**.

## Part 1 - capture route and tiers
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Route 2: one-page stock-capture protocol | `docs/CAPTURE_PROTOCOL.md` | protocol for LiDAR (Stray Scanner), video and photo | Done |
| Three input tiers, same output contract | `run_capture.py`, `brynx/lidar.py`, `brynx/rgb.py`, `brynx/photo.py` | tier auto-detected; one `plan.json` schema | Done |
| Photo tier: per-room folders -> stitched plan | `brynx/rgb.py` (`photo_stitch`), `brynx/photo.py` | rooms joined through doorway photos; unlinked rooms flagged | Partial (stitching needs doorway photos) |
| Video tier from any iPhone 15+ | `brynx/rgb.py` (`video_capture`), `brynx/mono.py` | MoGe-2 metric depth + MapAnything poses | Partial (accuracy below LiDAR; see benchmark) |
| LiDAR tier | `brynx/capture.py`, `brynx/lidar.py` | Stray Scanner -> plan | Done |
| Device matrix | `docs/DEVICE_MATRIX.md` | tier x device x accuracy x run time | Done |

## Part 2 - output contract and gates
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| Dimensioned per-room plan (walls, ceiling, area, openings) | `brynx/lidar.py` | `rooms[].dimensions`, `ceiling_height`, `area_m2`, `openings[]` | Done |
| Stitched multi-room plan, adjacency | `brynx/cells.py`, `brynx/spaces.py`, `brynx/render.py` | `adjacency`, `plan.png` | Done (LiDAR, video); Partial (photo) |
| Per-surface damage regions, class + metric extent | `brynx/damage.py`, `models/defect_probe.npz` | `damage.regions[]` (surface, room, m2 +- CI) | Done |
| Concealed-damage flags with rule fired | `brynx/damage.py` (`concealed_flags`) | `concealed_damage_flags[]` (R1-R6) | Done |
| Scope line items keyed to surfaces | `brynx/damage.py` (`scope_items`) | `scope[]` | Done |
| CI on every measurement | `brynx/lidar.py` (error model), `brynx/rgb.py` (tier error models) | `ci95` fields; `error_model` | Done |
| One command per capture, JSON, rendered plan | `run_capture.py`, `brynx/render.py` | `outputs/<name>/plan.json`, `plan.png` | Done |
| Openings <= 2 cm on >= 85 % | `benchmark/` | needs laser GT of openings | Missing (no opening GT) |
| Ceiling <= 1.5 cm per room; spread <= 1 cm | `brynx/lidar.py` (`room_ceiling`), `tools/ceiling_view.py` | ceiling reported only where seen; tape 2.718 m for home Room2 | Partial |
| Repeatability <= 1 cm / 0.5 % per wall | `tools/repeatability.py` | `docs/fixloop/*_rooms.txt` | Partial (17.2 cm median: fails) |
| Drift accountability, ablation on/off | `brynx/drift.py`, `tools/drift_ablation.py` | `outputs/<capture>/drift_ablation.png/.json` | Done |
| Photo-tier whole-property stitch, footprint +-8 % | `brynx/photo.py`, `tools/make_photo_set.py` | photo sets cut from LiDAR captures, LiDAR as truth | Partial |
| Photo +-8 % / video +-3 % wall lengths, calibrated | `tools/rgb_vs_lidar.py`, `benchmark/eval_tape.py` | see `docs/BENCHMARK.md` | Partial |

## Part 3 - head-to-head
| Requirement | File path | Artifact | Status |
|---|---|---|---|
| LiDAR tier vs Polycam / magicplan on 2 rooms | - | needs an iPhone Pro + app export | Missing (no Pro device available) |

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
| Reproduction bundle (cached model outputs replay; live path runs) | `outputs/<name>/cache/`, `--live` flag | Done |
| Benchmark report | `docs/BENCHMARK.md` | see file |
| Fix loop bundle | `docs/fixloop/` | Done |
| Technical report <= 6 pages | `docs/TECHNICAL_REPORT.md` | see file |
| Raw benchmark data | captures `c00a170fe1`, `1a8384c3f6`, `c7d28f72c6`, `data/home/`, `benchmark/gt/` | Partial (laser GT only for home Room2) |
