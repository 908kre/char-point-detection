import typing as t
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from cytoolz.curried import sorted, pipe
import os
from datetime import datetime
from mlboard_client import Writer

MLBOARD_URL = os.getenv("MLBOARD_URL", "http://192.168.10.8:2020")


param = {
    "boosting": "dart",
    "objective": "regression",
}
w = Writer(MLBOARD_URL, f"{datetime.now()}", {
    "ignore_columns":["用途", "種類", "土地の形状", "市区町村コード", "今後の利用目的", "id", "地域", "取引の事情等", "建ぺい率（％）", "都市計画", "建物の構造"]
},)


def eval(preds, train_data):
    metric = mean_squared_error(train_data.get_label(), preds, squared=False)
    w.add_scalars({"rmse": metric})
    return metric, 2, 3


def create_dataset(df: t.Any) -> t.Tuple[t.Any, t.Sequence[str]]:
    y = df["y"]
    x = df.drop("y", axis=1)
    return lgb.Dataset(x.values, label=y.values), x.columns


def train(train_path: str, test_path: str) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_data, columns = create_dataset(train_df)
    test_data, _ = create_dataset(test_df)
    bst = lgb.train(
        param,
        train_data,
        valid_sets=[test_data],
        valid_names=["Test"],
        num_boost_round=1000,
        early_stopping_rounds=10,
        feval=eval,
    )
    ranks = pipe(
        zip(bst.feature_importance(), columns), sorted(key=lambda x: -x[0]), list,
    )
    print(ranks)
