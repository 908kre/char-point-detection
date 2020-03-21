import typing as t
from sklearn.metrics import f1_score


def eval(preds: t.Any, labels: t.Any) -> float:
    return f1_score(labels, preds, average="macro")
