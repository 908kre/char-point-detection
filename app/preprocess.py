import typing as t
from cytoolz.curried import groupby, valmap, pipe, unique, map
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from app import config
from app.entities import BBox, BBoxes, Images, Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_bboxes(bboxes: BBoxes, path: str) -> None:
    fig, axs = plt.subplots(4, sharex=False)
    sizes = [b.size() for b in bboxes]
    axs[0].hist([b.w for b in bboxes], bins=100)
    axs[0].set_ylabel("w")
    axs[1].hist([b.h for b in bboxes], bins=100)
    axs[1].set_ylabel("h")

    axs[2].hist([b.h / b.w for b in bboxes], bins=1000)
    axs[2].set_ylabel("aspect")
    blank_img = np.zeros((1024, 1024))
    axs[3].imshow(blank_img)
    for bbox in bboxes:
        rect = mpatches.Rectangle(
            (0, 0), bbox.w, bbox.h, fill=False, edgecolor="red", linewidth=1,
        )
        axs[3].add_patch(rect)
    plt.savefig(Path(config.plot_dir).joinpath(path))
    plt.close()


def to_bbox(value: str) -> BBox:
    arr = np.fromstring(value[1:-1], sep=",")
    return BBox(*arr)


def load_lables(limit: t.Union[None, int] = None) -> Images:
    df = pd.read_csv(config.label_path, nrows=limit)
    rows = df.to_dict("records")
    images = pipe(
        rows,
        groupby(lambda x: x["image_id"]),
        valmap(
            lambda x: Image(
                id=x[0]["image_id"],
                source=x[0]["source"],
                width=x[0]["width"],
                height=x[0]["height"],
                bboxes=[to_bbox(b["bbox"]) for b in x],
            )
        ),
    )
    return images


def plot_with_bbox(image: Image) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    image_arr = image.get_arr()
    ax.imshow(image_arr)
    for bbox in image.bboxes:
        rect = mpatches.Rectangle(
            (bbox.x, bbox.y), bbox.w, bbox.h, fill=False, edgecolor="red", linewidth=1,
        )
        ax.add_patch(rect)
    plt.savefig(Path(config.plot_dir).joinpath(f"bbox-{image.id}.jpg"))
    plt.close()


class KFold:
    def __init__(self, n_splits: int = config.n_splits):
        self._skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=config.random_state
        )

    def __call__(self, images: Images) -> t.Iterator[t.Tuple[Images, Images]]:
        rows = list(images.values())
        fold_keys = pipe(rows, map(lambda x: f"{x.source}-{len(x.bboxes) // 1}"), list)
        for train, valid in self._skf.split(X=rows, y=fold_keys):
            train_rows = {rows[i].id: rows[i] for i in train}
            valid_rows = {rows[i].id: rows[i] for i in valid}
            yield (train_rows, valid_rows)
