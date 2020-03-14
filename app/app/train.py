import typing as t
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

def eval(preds, train_data):
    metric = mean_squared_error(train_data.get_label(), preds, squared=False)
    return metric, 2, 3

def create_dataset(df:t.Any) -> t.Any:
    y = df['y']
    x = df.drop('y', axis=1)
    print(x.columns)
    print(x.values)
    return lgb.Dataset(x.values, label=y.values)

def train(train_path:str, test_path:str) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_data = create_dataset(train_df)
    test_data = create_dataset(test_df)
    param = {
        'boosting': 'dart',
        'objective': 'regression',
        'learning_rate': 0.01,
    }
    lgb.train(
        param,
        train_data,
        valid_sets=[train_data, test_data],
        valid_names=['Train', 'Test'],
        num_boost_round=10,
        early_stopping_rounds=100,
        feval=eval,
    )
