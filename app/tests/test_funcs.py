import typing as t
from app.eval import eval
from cytoolz.curried import pipe, sliding_window, keyfilter
import numpy as np


def test_eval() -> None:
    preds = [1, 2, 3, 4]
    labels = [1, 2, 3, 4]
    res = eval(preds, labels,)
    assert res == 1.0


def test_markov_chain() -> None:
    trajectory = [0, 1, 1, 0, 0, 0]
    trajectory_count: t.Dict[t.Tuple[int, int], int] = {}
    for trans in sliding_window(2)(trajectory):
        if trans in trajectory_count:
            trajectory_count[trans] += 1
        else:
            trajectory_count[trans] = 1
    n_states = len(set(trajectory))
    transition_matrix = np.empty([n_states, n_states])
    for k, v in trajectory_count.items():
        total_transition = sum(
            keyfilter(lambda x: x[0] == k[0])(trajectory_count).values()
        )
        transition_matrix[k[0], k[1]] = v / total_transition

    assert transition_matrix.tolist() == [[2 / 3, 1 / 3], [1 / 2, 1 / 2]]
