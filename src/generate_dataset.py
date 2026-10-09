import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import robosuite as suite
import robosuite.macros as macros

from PIL import Image, ImageDraw
from robosuite.utils.camera_utils import get_camera_transform_matrix

from compat_robosuite import apply_patch

OBJECTS = ("Milk", "Bread", "Cereal", "Can")
CAMERA = "agentview"
SIZE=512

def project_to_image(position, transform, height, width):
    point = np.append(position, 1.0)
    projected = transform @ point
    depth = projected[2]
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("Object center is behind the camera")

    x = projected[0] / depth
    y = projected[1] / depth

    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Object center outside image: {(x, y)}")

    pixel_xy = [float(x), float(y)]
    normalized_xy = [
        float(x / width),
        float(y / height)
    ]

    return pixel_xy, normalized_xy

def make_preview(image, labels, output_path):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)

    for label in labels:
        x, y = label["pixel_xy"]
        draw.ellipse(
            (x-5, y-5, x+5, y+5),
            fill="red",
            outline="white",
            width=2,
        )

        draw.text(
            (x+9, y+7),
            label["object"],
            fill="red",
            stroke_width=2,
            stroke_fill="white",
        )

    preview.save(output_path)

def generate(args):
    apply_patch()

    macros.IMAGE_CONVENTION = "opencv"
    root = Path(args.out)
    
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(args.scenes))
    random.Random(args.seed).shuffle(indices)

    train_end = int(0.8 * args.scenes)
    val_end = train_end + int(0.1 * args.scenes)

    split_map = {}
    for i, scene_id in enumerate(indices):
        if i < train_end:
            split_map[scene_id] = "train"
        elif i < val_end:
            split_map[scene_id] = "val"
        else:
            split_map[scene_id] = "test"

    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=CAMERA,
        camera_heights=SIZE,
        camera_widths=SIZE,
        single_object_mode=0,
        hard_reset=False,
        seed=args.seed,
    )

    records = defaultdict(list)

    try:
        for scene_id in range(args.scenes):
            obs = env.reset()
            image_array = obs[f"{CAMERA}_image"]
            image = Image.fromarray(
                np.asarray(image_array, dtype=np.uint8)
            )
            height, width = image_array.shape[:2]
            camera_matrix = get_camera_transform_matrix(env.sim, CAMERA, height, width)
            labels = []
            for object_name in OBJECTS:
                position = obs[f"{object_name}_pos"]
                pixel_xy, target_xy = project_to_image(position, camera_matrix, height, width)
                labels.append({
                    "object": object_name,
                    "pixel_xy": pixel_xy,
                    "target_xy": target_xy,
                })

            filename = f"scene_{scene_id:05d}.png"
            image.save(image_dir / filename)

            if scene_id < args.previews:
                make_preview(image, labels, preview_dir / filename)

            for label in labels:
                records[split_map[scene_id]].append({
                    "scene_id": scene_id,
                    "image": f"images/{filename}",
                    "task": f"pick {label['object'].lower()}",
                    "target_xy": label["target_xy"],
                })

            print(
                f"Scene {scene_id+1}/{args.scenes} "
                f"{len(labels)} paired samples"
            )

    finally:
        env.close()

    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        with path.open("w") as file:
            for record in records[split]:
                file.write(json.dumps(record) + "\n")

        print(f"{split}: {len(records[split])} samples")

    print(f"Dataset saved to: {root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--scenes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--previews", type=int, default=5)
    parser.add_argument("--out", default="data/paired")

    args = parser.parse_args()

    if args.scenes < 1:
        parser.error("--scenes must be at least 1")

    generate(args)
    