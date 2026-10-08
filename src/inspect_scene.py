from pathlib import Path

import numpy as np 
import robosuite as suite
from PIL import Image

from compat_robosuite import apply_patch

apply_patch()


def main():
    env = suite.make(
        env_name="PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        single_object_mode=0,
    )

    try:
        obs = env.reset()
        
        print("Available observations:")
        for key, value in obs.items():
            print(f"    {key}: {value.shape}")

        image = obs["agentview_image"]

        output = Path("results")
        output.mkdir(exist_ok=True)

        Image.fromarray(np.flipud(image)).save(output / "first_scene.png")
        print("Saved results/first_scene.png")

    finally:
        env.close()

if __name__ == "__main__":
    main()
