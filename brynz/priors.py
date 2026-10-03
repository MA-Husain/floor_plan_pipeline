"""Architectural and sensor priors - every structural threshold in the pipeline lives here, with
its justification. None of these are fitted to the benchmark captures; they come from building
norms (residential doors / corridors / room minimums) and the iPhone LiDAR spec.

Changing a value here is a design decision and must be re-scored on ALL captures
(benchmark/eval_spaces.py), including held-out ones.
"""

# --- openings (residential norms: internal door leaves 0.6-0.9 m; clear opening incl. frame up to
#     ~1.0 m; double / pocket doors to 1.2-1.3 m)
DOOR_MIN_M = 0.55        # narrower gaps are clutter between furniture faces
DOOR_MAX_M = 1.30        # wider gaps are open-plan passages, not doors
WINDOW_MAX_M = 3.0       # glazing runs wider than this are treated as missing wall

# --- circulation (corridor clear width 0.9-1.2 m typical, <=1.6 m generous)
CORRIDOR_MAX_WIDTH_M = 1.6
CORRIDOR_MIN_ELONGATION = 2.0
NECK_MAX_M = 2.0         # a passage <= this between two areas separates them as distinct spaces
NECK_MIN_SIDE_M2 = 3.0   # each side must be at least a small room to count as a separate space

# --- spaces (smallest habitable / sanitary spaces: WC ~1.2-1.5 m2; closet ~0.8-2.5 m2)
SPACE_MIN_AREA_M2 = 1.0
SPACE_MIN_WIDTH_M = 0.7  # narrower regions are slivers (wall interiors, gaps behind furniture)
CLOSET_MAX_AREA_M2 = 2.8

# --- walls (furniture rarely exceeds ~1.2 m except wardrobes / fridges)
WALL_MIN_HEIGHT_COVERAGE = 0.55   # fraction of the observed height band a wall face must cover
WALL_MIN_SOLID_M = 0.4

# --- sensor (Apple LiDAR: reliable to ~4-5 m; confidence 2 = high)
MAX_DEPTH_M = 4.0
MIN_CONFIDENCE = 2

# --- ceiling: a measured ceiling plane must span a real part of the room. Shelf / wardrobe /
#     stair-soffit undersides also face down but cover only a small fraction of the plan.
CEILING_MIN_COVERAGE = 0.5      # below this the ceiling is reported as not measured
