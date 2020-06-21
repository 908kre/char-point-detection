import typing as t
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from app.entities import Sample
from cytoolz.curried import groupby, valmap, pipe, unique, map


class KFold:
    def __init__(self, n_splits: int):
        self._skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    def __call__(
        self, rows: t.List[Sample]
    ) -> t.Iterator[t.Tuple[t.List[Sample], t.List[Sample]]]:
        fold_keys = pipe(rows, map(lambda x: f"{len(x[1])}"), list)
        for train, valid in self._skf.split(X=rows, y=fold_keys):
            yield ([rows[i] for i in train], [rows[i] for i in valid])
