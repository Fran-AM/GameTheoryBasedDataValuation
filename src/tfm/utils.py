"""This module contains utility functions for the project."""
import os
import random
import logging
from typing import Iterable, List
from pydvl.utils.dataset import Dataset


import numpy as np
import pandas as pd

from itertools import chain, combinations
from sklearn.linear_model import LogisticRegression

__all__ = [
    "set_random_seed",
    "setup_logger",
    "equilibrate_clases",
    "exact_banzhaf",
]

# TODO: Set seed for cuda
def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    #torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logger():
    """
    Setup the logger for the project.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    return logger

def build_pyDVL_dataset(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dataset:
    """
    Build a pyDVL dataset from numpy arrays.
    Flatts the images for pyDVL compatibility.

    Args:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The test labels.
    
    Returns:
        Dataset: The pyDVL dataset.
    """
    shape = X_train.shape
    # If 4D array (batch_size, width, height, depth/channels)
    if len(shape) == 4:
        batch_size, w, h, p = shape
        X_train = X_train.reshape(batch_size, w * h * p)
    # If 3D array (batch_size, width, height)
    elif len(shape) == 3:
        batch_size, w, h = shape
        X_train = X_train.reshape(batch_size, w * h)
    else:
        raise ValueError("Something is wrong with dimensions")

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return Dataset(
        x_train=X_train.numpy(),
        y_train=y_train.numpy(),
        x_test=X_test.numpy(),
        y_test=y_test.numpy(),
    )

def equilibrate_clases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Equilibrate the classes of a dataset with
    two classes in the target variable.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    target_counts = df['target'].value_counts()
    min_count = target_counts.min()
    
    balanced = pd.concat([
        df[df['target'] == class_label].sample(min_count)
        for class_label in target_counts.index
    ], axis=0)
    
    return balanced


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

