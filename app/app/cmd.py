import yaml
import argparse
import pandas as pd
from .encoders import parse_floor
from cytoolz.curried import map, pipe

parser = argparse.ArgumentParser(description="Process some integers.")
funcs = {"parse_floor": parse_floor}


def parse_config(fpath: str) -> dict:
    with open(r"config.yml") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def main():
    parser.add_argument("--config", type=str, help="config")
    args = parser.parse_args()
    config = parse_config(args.config)
    df = pd.read_csv(config["dataset"]["train"])
    ignore_columns = config["preprocess"]["ignore"]
    df = df.drop(ignore_columns, axis="columns")
    for i, r in df.iterrows():
        print(i)
        #  print(r["間取り", "面積"])


def eda():
    parser.add_argument("--config", type=str, help="config")
    args = parser.parse_args()
    config = parse_config(args.config)
    df = pd.read_csv(config["dataset"]["train"])
    #  print(df.describe())
    print(df.isnull().sum() / len(df))
    print(df["面積（㎡）"].unique())
