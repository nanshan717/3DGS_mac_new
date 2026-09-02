#!/usr/bin/env python3
"""Pre-registered CoffeeFabric-Syn P03/P04 scene designs.

This module intentionally has no Blender dependency so the held-out geometry
specification can be unit-tested before any reconstruction is trained.
"""

from __future__ import annotations

import math


HELDOUT_PROFILES = {
    "p03": {
        "scene_id": "P03_sloped_high_occlusion_v7_heldout96",
        "seed": 303,
        "role": "held_out",
        "morphology": "cross_slope_with_transverse_wrinkles",
        "occlusion_level": "high",
        "plant_positions": (
            (-1.92, -2.78), (-0.70, -2.54), (0.55, -2.82), (1.84, -2.50),
            (-1.64, -1.05), (-0.35, -0.78), (0.92, -1.12), (1.96, -0.82),
            (-1.90, 0.60), (-0.62, 0.94), (0.71, 0.62), (1.75, 0.98),
            (-1.70, 2.44), (-0.42, 2.18), (0.82, 2.55), (1.92, 2.28),
        ),
        "plant_height_range": (0.62, 0.92),
        "weed_patch_count": 96,
        "interior_weed_stride": 3,
        "lighting": "overcast_soft",
    },
    "p04": {
        "scene_id": "P04_piecewise_mixed_occlusion_v7_heldout96",
        "seed": 404,
        "role": "held_out",
        "morphology": "piecewise_crossfall_with_local_uplift_and_edge_sag",
        "occlusion_level": "mixed",
        "plant_positions": (
            (-1.78, -2.66), (-0.18, -2.42), (1.57, -2.73),
            (-1.91, -1.13), (-0.58, -0.67), (0.82, -1.04), (1.89, -0.71),
            (-1.66, 0.58), (-0.20, 0.96), (1.36, 0.69),
            (-1.93, 2.33), (-0.72, 2.58), (0.61, 2.24), (1.82, 2.52),
        ),
        "plant_height_range": (0.44, 0.86),
        "weed_patch_count": 68,
        "interior_weed_stride": 5,
        "lighting": "oblique_hard_shadow",
    },
}


def heldout_height(profile: str, x: float, y: float) -> float:
    """Return the deterministic scene-level support height in metres."""
    if profile == "p03":
        # A cross-row grade plus low-amplitude transverse wrinkles.  The
        # frequencies/phases are fixed here before any P03 result is observed.
        return (0.043 * x - 0.011 * y
                + 0.038 * math.sin(0.78 * y + 0.45)
                + 0.019 * math.sin(1.55 * x - 0.52 * y + 0.20))
    if profile == "p04":
        # Continuous piecewise crossfall, a localized fabric uplift and a
        # shallow edge sag.  This is deliberately unlike P01/P02/P03.
        crossfall = 0.026 * x if x < 0.0 else -0.018 * x
        ridge = 0.052 * math.exp(-((x - 0.72) / 0.62) ** 2 - ((y + 0.35) / 1.15) ** 2)
        edge_sag = -0.036 * math.exp(-((x + 2.22) / 0.34) ** 2 - ((y - 1.55) / 1.05) ** 2)
        return crossfall + ridge + edge_sag + 0.014 * math.sin(0.92 * x + 0.67 * y)
    raise ValueError(f"Not a held-out scene profile: {profile}")


def validate_profiles() -> None:
    """Fail fast if a future edit weakens the pre-registered split contract."""
    if set(HELDOUT_PROFILES) != {"p03", "p04"}:
        raise ValueError("The frozen held-out release requires exactly P03 and P04")
    seeds = {spec["seed"] for spec in HELDOUT_PROFILES.values()}
    scene_ids = {spec["scene_id"] for spec in HELDOUT_PROFILES.values()}
    if len(seeds) != 2 or len(scene_ids) != 2:
        raise ValueError("Held-out seeds and scene IDs must be unique")
    for name, spec in HELDOUT_PROFILES.items():
        if spec["role"] != "held_out" or len(spec["plant_positions"]) < 12:
            raise ValueError(f"Invalid held-out contract for {name}")


validate_profiles()
