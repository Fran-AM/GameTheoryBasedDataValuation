from typing import Iterable, List

import numpy as np
import pandas as pd

from itertools import chain, combinations
from sklearn.linear_model import LogisticRegression

def _powerset(iterable: Iterable) -> Iterable:
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def _utility(X: pd.DataFrame, y: np.ndarray, subset: List) -> float:
    clf = LogisticRegression(solver='sag')
    # Puede que haya que añadir parametro validation_fraction.
    if len(np.unique(y[subset])) < 2:
        return 0
    clf.fit(X.iloc[subset], y[subset])
    # Usamos todo el dataset para calcular el score.
    return clf.score(X, y)

def exact_banzhaf(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    n = len(X)
    banzhaf_values = np.zeros(n)
    for i in range(n):
        subsets = list(_powerset([j for j in range(n) if j != i]))
        for subset in subsets:
            S_i = list(subset) + [i]
            banzhaf_values[i] += (_utility(X, y, S_i) - _utility(X, y, list(subset)))
        banzhaf_values[i] /= (2 ** (n - 1))
    return banzhaf_values

# Hay que hacer algún test para ver que funciona bien.