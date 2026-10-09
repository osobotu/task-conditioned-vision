
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_positions(dataset_dir):
    """Load object-center coordinates from all splits."""
    positions = defaultdict(list)

    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.jsonl"

        with path.open() as f:
            for line in f:
                sample = json.loads(line)

                object_name = sample["task"].removeprefix("pick ")
                positions[object_name].append(sample["target_xy"])

    return {
        name: np.array(coords)
        for name, coords in positions.items()
    }


def plot_position_distributions(positions, output_path, bins=8):
    """Plot position scatter and spatial density for each object."""

    objects = sorted(positions)
    fig, axes = plt.subplots(
        2, len(objects),
        figsize=(4 * len(objects), 8),
        constrained_layout=True,
    )

    for i, name in enumerate(objects):
        points = positions[name]
        x = points[:, 0]
        y = points[:, 1]

        # Row 1: Scatter plot
        ax = axes[0, i]
        ax.scatter(x, y, alpha=0.55, s=18)

        ax.set_title(f"{name.capitalize()} — positions")
        ax.set_xlabel("Normalized x")
        ax.set_ylabel("Normalized y")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)  # Image origin is top-left
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

        # Row 2: Spatial density heatmap
        counts, x_edges, y_edges = np.histogram2d(
            x, y,
            bins=bins,
            range=[[0, 1], [0, 1]],
        )

        ax = axes[1, i]
        heatmap = ax.pcolormesh(
            x_edges,
            y_edges,
            counts.T,
            cmap="viridis",
            shading="auto",
            vmin=0,
        )

        ax.set_title(f"{name.capitalize()} — density")
        ax.set_xlabel("Normalized x")
        ax.set_ylabel("Normalized y")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect("equal")

        fig.colorbar(
            heatmap,
            ax=ax,
            label="Scene count",
            shrink=0.75,
        )

        occupied = np.count_nonzero(counts)
        total = counts.size

        print(f"\n{name.upper()}")
        print(f"Samples: {len(points)}")
        print(f"X range: {x.min():.3f} to {x.max():.3f}")
        print(f"Y range: {y.min():.3f} to {y.max():.3f}")
        print(f"X std:   {x.std():.3f}")
        print(f"Y std:   {y.std():.3f}")
        print(f"Grid cells occupied: {occupied}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    positions = load_positions(Path("data/paired"))

    plot_position_distributions(
        positions,
        Path("results/position_distributions.png"),
    )
