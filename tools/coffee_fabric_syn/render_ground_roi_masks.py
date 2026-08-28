#!/usr/bin/env python3
"""Render visible ground/fabric ROI masks from a frozen CoffeeFabric Blender scene."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy


def args_after_separator():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("scene")
    parser.add_argument("--resolution", type=int, default=800)
    return parser.parse_args(argv)


def emission_material(name, value):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (value, value, value, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = args_after_separator()
    root = Path(args.scene).expanduser().resolve()
    plan = json.loads((root / "camera_plan.json").read_text(encoding="utf-8"))
    output = root / "bsr_masks"
    output.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False

    scene.world.use_nodes = True
    background = next(node for node in scene.world.node_tree.nodes if node.type == "BACKGROUND")
    background.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs["Strength"].default_value = 0.0

    black = emission_material("roi_excluded_black", 0.0)
    white = emission_material("roi_ground_white", 1.0)
    included = {"terrain", "fabric_gt"}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        obj.hide_render = False
        obj.data.materials.clear()
        obj.data.materials.append(white if obj.name in included else black)

    rendered = []
    for spec in plan:
        camera = bpy.data.objects.get(f"dataset_camera_{spec['frame']:05d}")
        if camera is None:
            raise RuntimeError(f"Missing camera dataset_camera_{spec['frame']:05d}")
        scene.camera = camera
        path = output / f"{spec['frame']:05d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append(path)

    manifest = root / "bsr_masks_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "coffee-fabric-visible-ground-roi-v1",
        "source_scene": root.name,
        "included_objects": sorted(included),
        "excluded_classes": ["coffee", "weeds", "posts", "root_guards", "sky"],
        "count": len(rendered),
        "resolution": [args.resolution, args.resolution],
        "threshold": 0.5,
        "files": [str(path.relative_to(root)) for path in rendered],
    }, indent=2) + "\n", encoding="utf-8")

    checksum_file = root / "SHA256SUMS"
    existing = []
    if checksum_file.exists():
        for line in checksum_file.read_text(encoding="utf-8").splitlines():
            _, relative = line.split(maxsplit=1)
            if not relative.startswith("bsr_masks/") and relative != manifest.name:
                existing.append(relative)
    tracked = [root / relative for relative in existing] + [manifest, *rendered]
    checksum_file.write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(root)}" for path in tracked) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"scene": str(root), "masks": len(rendered), "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
