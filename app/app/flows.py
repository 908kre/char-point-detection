import pandas as pd
from .encoders import (
    parse_duration,
    parse_age,
    parse_quater,
    parse_floor,
    parse_erea,
    label_encode,
    fillna_mean,
)


def preprocess(input_path:str, output_path:str) -> None:
    df = pd.read_csv(input_path)
    ignore_columns = ["用途", "土地の形状", "市区町村コード", "今後の利用目的"]
    df = df.drop(ignore_columns, axis="columns")

    for c in df.columns:

        if c in [
            "種類",
            "地域",
            "都道府県名",
            "市区町村名",
            "地区名",
            "最寄駅：名称",
            "建物の構造",
            "前面道路：方位",
            "前面道路：種類",
            "都市計画",
            "間口",
            "改装",
        ]:
            df[c] = label_encode(df[c], cache_dir="/tmp", key=c,)

        if c == "最寄駅：距離（分）":
            df[c] = df[c].apply(parse_duration)
            df[c] = fillna_mean(df[c])

        if c == "取引時点":
            df = pd.concat(
                [df, pd.DataFrame(df[c].apply(parse_quater).tolist())], axis=1
            )
            df = df.drop([c], axis=1)

        if c == "間取り":
            df = pd.concat(
                [df, pd.DataFrame(df[c].apply(parse_floor).tolist())], axis=1
            )
            df = df.drop([c], axis=1)

        if c in ["面積（㎡）", "延床面積（㎡）"]:
            df[c] = df[c].apply(parse_erea)

        if c == "建築年":
            df[c] = df[c].apply(parse_age)
    df.to_csv(output_path)
