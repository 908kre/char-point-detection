from app import config
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from app.models.efficientdet import Anchors


def test_anchors() -> None:
    scales = [1.0]
    ratios = [1.0]

    images = torch.ones((1, 1, 32, 32))
    fn = Anchors(pyramid_levels=[1], scales=scales, ratios=ratios, stride=4)
    res = fn(images)

    _, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(images[0, 0])
    unit = len(scales) * len(ratios)

    ax.scatter(
        (res[0][:, 0] + res[0][:, 2]) / 2,
        (res[0][:, 1] + res[0][:, 3]) / 2,
        s=10,
        marker=".",
        c="r",
    )
    for offset, color in [(0, "red"), (8, "blue")]:
        for bbox in res[0][unit * offset : unit * (offset + 1)]:
            bbox = bbox.numpy()
            rect = mpatches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                fill=False,
                edgecolor=color,
                linewidth=1,
            )
            ax.add_patch(rect)

    plt.savefig(Path(config.plot_dir).joinpath(f"test-anchor.png"))
    print(res)
    #  print(res.shape)
    #  assert res.shape == (1, 9 * 4, 4)
