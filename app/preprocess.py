from cytoolz.curried import groupby, valmap, pipe
import pandas as pd
import numpy as np
from pathlib import Path
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.io import imread
from sklearn.model_selection import StratifiedKFold
from app import config
from app.entities import BBox, BBoxs, Images, Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def to_bbox(value: str) -> BBox:
    arr = np.fromstring(value[1:-1], sep=",")
    return BBox(*arr)


def load_lables() -> Images:
    df = pd.read_csv(config.label_path)
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
                bboxs=[to_bbox(b["bbox"]) for b in x],
            )
        ),
    )
    return images


def plot_with_bbox(image: Image) -> None:
    image_path = Path(config.train_dir).joinpath(f"{image.id}.jpg")
    image_arr = imread(image_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.grid(False)
    ax.imshow(image_arr)
    for bbox in image.bboxs:
        rect = mpatches.Rectangle(
            (bbox.x, bbox.y), bbox.w, bbox.h, fill=False, edgecolor="red", linewidth=1,
        )
        ax.add_patch(rect)
    plt.savefig(Path(config.plot_dir).joinpath(f"bbox-{image.id}.jpg"))
    plt.close()


#
#
#  def kfold(bboxs: BBoxs) -> None:
#      skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
#      group = groupby(lambda x: x["image_id"])(bboxs)
#      image_ids = list(group.keys())
#      kfold_keys = valmap(lambda x: f"{x[0]['source']}-{len(x)}")(group)
#      print(kfold_keys)
