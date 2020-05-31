import seaborn as sns
import pandas as pd
from app import config

sns.set()


def load_lables() -> None:
    df = pd.read_csv(config.label_path)[:10]
    for r in df.iterrows():
        print(r)
    print(len(df))


def plot_with_bbox(image_id: str) -> None:
    print(image_id)
    #  plt.imshow(image_id)
