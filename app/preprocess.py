import pandas as pd
from pathlib import Path
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from skimage.io import imread
from app import config
from app.entities import TrainLabel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_lables() -> None:
    df = pd.read_csv(config.label_path)[:10]
    for r in df.iterrows():
        print(r)
    print(len(df))


def plot_with_bbox(label: TrainLabel) -> None:
    image_path = Path(config.train_dir).joinpath(f"{label.image_id}.jpg")
    image = imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(False)
    ax.imshow(image)
    rect = mpatches.Rectangle(
        (label.bbox[0], label.bbox[1]),
        label.bbox[2],
        label.bbox[3],
        fill=False,
        edgecolor="red",
        linewidth=1,
    )
    ax.add_patch(rect)
    plt.savefig(Path(config.plot_dir).joinpath("bbox-{label.image_id}.jpg"))
