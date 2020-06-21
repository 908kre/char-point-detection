import typing as t
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


def kfold(
    n_splits: int, keys: t.List[t.Any],
) -> t.Iterator[t.Tuple[t.List[int], t.List[int]]]:
    sk = StratifiedKFold(n_splits=n_splits, shuffle=True)
    return sk.split(X=range(len(keys)), y=keys)
