import numpy as np
import typing as t
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from skimage import io
import glob
from tqdm import tqdm
from .entities import Label, Labels, Annotations
from .dataset import Dataset
from cytoolz.curried import unique, pipe, map, mapcat, frequencies, topk
import seaborn as sns

sns.set()


def load_labels(path: str) -> Labels:
    df = pd.read_csv(path)
    rows: Labels = dict()
    for (idx, value) in df.iterrows():
        c, d = encode_attribute(value["attribute_name"])
        rows[idx] = {"id": idx, "category": c, "detail": d}
    return rows


def load_images(path: str, labels: Labels) -> t.Any:
    df = pd.read_csv(path)
    rows: t.Dict[int, Label] = dict()
    for (idx, value) in df.iterrows():
        c, d = encode_attribute(value["attribute_name"])
        rows[idx] = {"id": idx, "category": c, "detail": d}
    return rows


def encode_attribute(name: str) -> t.Tuple[str, str]:
    splited = name.split("::")
    return splited[0], splited[1]


def get_annotations(path: str, labels: Labels) -> Annotations:
    df = pd.read_csv(path)
    df["attribute_ids"] = df["attribute_ids"].apply(
        lambda x: [int(i) for i in x.split(" ")]
    )
    annotations: Annotations = []
    for _, vs in df.iterrows():
        annotations.append(
            {"id": vs["id"], "label_ids": vs["attribute_ids"],}
        )
    return annotations


def get_images_summary(image_dir: str) -> t.Dict:
    paths = glob.glob(os.path.join(image_dir, "*.png"))
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as e:
        shapes = np.zeros((len(paths), 2))
        gray_count = 0
        rgb_count = 0
        for i, (p, img) in tqdm(enumerate(zip(paths, e.map(io.imread, paths)))):
            if len(img.shape) == 3:
                rgb_count += 1
            else:
                gray_count += 1
            shapes[i] = np.array(img.shape[:2])
        aspect = shapes[:, 1] / shapes[:,0]
    return {
        "gray_count": gray_count,
        "rgb_count": rgb_count,
        "min_aspect": aspect.min(),
        "mean_aspect": aspect.mean(),
        "max_aspect": aspect.max(),
        "min_width": shapes[:, 0].min(),
        "mean_width": shapes[:, 0].mean(),
        "max_width": shapes[:, 0].max(),
        "min_height": shapes[:, 1].min(),
        "mean_height": shapes[:, 1].mean(),
        "max_height": shapes[:, 1].max(),
    }


def get_summary(annotations: Annotations, labels: Labels) -> t.Any:
    count = len(annotations)
    label_count = pipe(annotations, map(lambda x: len(x["label_ids"])), list, np.array)
    label_hist = {
        5: np.sum(label_count == 5),
        4: np.sum(label_count == 4),
        3: np.sum(label_count == 3),
    }

    label_ids = pipe(annotations, mapcat(lambda x: x["label_ids"]), list, np.array,)
    total_label_count = len(label_ids)
    top3 = pipe(
        frequencies(label_ids).items(),
        topk(3, key=lambda x: x[1]),
        map(lambda x: (f"{labels[x[0]]['category']}::{labels[x[0]]['detail']}", x[1],)),
        list,
    )

    worst3 = pipe(
        frequencies(label_ids).items(),
        topk(3, key=lambda x: -x[1]),
        map(lambda x: (f"{labels[x[0]]['category']}::{labels[x[0]]['detail']}", x[1],)),
        list,
    )
    return {
        "count": count,
        "label_hist": label_hist,
        "label_count_mean": label_count.mean(),
        "label_count_median": np.median(label_count),
        "label_count_max": label_count.max(),
        "label_count_min": label_count.min(),
        "total_label_count": total_label_count,
        "top3": top3,
        "worst3": worst3,
    }
