import yaml
import argparse
import pandas as pd
from .encoders import parse_floor, label_encode, fillna_mean, parse_duration, parse_erea, parse_age, parse_quater
from cytoolz.curried import map, pipe

parser = argparse.ArgumentParser(description="Process some integers.")


def parse_config(fpath: str) -> dict:
    with open(fpath) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def preprocess() -> None:
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="input file")
    parser.add_argument(
        "--config", type=str, help="config file", default="preprocess.yml"
    )
    args = parser.parse_args()
    config = parse_config(args.config)
    df = pd.read_csv(args.input)
    ignore_columns = [
        "用途",
        "土地の形状",
        "市区町村コード",
        "今後の利用目的"
    ]
    df = df.drop(ignore_columns, axis="columns")

    for c in df.columns:

        if c in ["種類", "地域", "都道府県名", "市区町村名", "地区名", "最寄駅：名称", "建物の構造", "前面道路：方位", "前面道路：種類", "都市計画", "間口", "改装",]:
            df[c] = label_encode(df[c], cache_dir=config["cache_dir"], key=c,)

        if c == "最寄駅：距離（分）":
            df[c] = df[c].apply(parse_duration)
            df[c] = fillna_mean(df[c])

        if c == "取引時点":
            df = pd.concat([df, pd.DataFrame(df[c].apply(parse_quater).tolist())], axis=1)
            df = df.drop([c], axis=1)

        if c == "間取り":
            df = pd.concat([df, pd.DataFrame(df[c].apply(parse_floor).tolist())], axis=1)
            df = df.drop([c], axis=1)

        if c in ["面積（㎡）", "延床面積（㎡）"]:
            df[c] =  df[c].apply(parse_erea)

        if c == "建築年":
            df[c] =  df[c].apply(parse_age)
        if c == "前面道路：幅員（ｍ）":
            print(df[c])
    df.to_csv(args.output)


def eda() -> None:
    parser.add_argument("--config", type=str, help="config")
    args = parser.parse_args()
    config = parse_config(args.config)
    df = pd.read_csv(config["dataset"]["train"])
    print(df.isnull().sum() / len(df))
    print(df["取引の事情等"].unique())
