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
    "make_balance_sample",
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

def _reshape_data(data: np.ndarray) -> np.ndarray:
    """
    Auxiliary function to reshape the data for pyDVL compatibility.

    Args:
        data (np.ndarray): The data to reshape.

    Returns:
        np.ndarray: The reshaped data.
    """
    shape = data.shape
    # If 4D array (batch_size, width, height, depth/channels)
    if len(shape) == 4:
        batch_size, w, h, p = shape
        return data.reshape(batch_size, w * h * p)
    # If 3D array (batch_size, width, height)
    elif len(shape) == 3:
        batch_size, w, h = shape
        return data.reshape(batch_size, w * h)
    else:
        raise ValueError("Something is wrong with dimensions")

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
    X_train = _reshape_data(X_train)
    X_test = _reshape_data(X_test)

    # Esto puede que no sea lo mejor tenerlo aquí
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return Dataset(
        x_train=X_train.numpy(),
        y_train=y_train.numpy(),
        x_test=X_test.numpy(),
        y_test=y_test.numpy(),
    )

def oversamp_equilibration(
        data: np.ndarray,
        target: np.ndarray
    )->(np.ndarray, np.ndarray):
    """
    Equilibrate the classes of a dataset with
    two classes in the target variable,
    using oversampling.

    Args:
        data (np.ndarray): The data.
        target (np.ndarray): The target.

    Returns:
        (np.ndarray, np.ndarray): The balanced data and target.
    """
    # Identify the minority class
    if np.mean(target) < 0.5:
        minor_class, major_class = 1, 0
    else:
        minor_class, major_class = 0, 1

    # Find the indices of the minority and majority classes
    index_minor_class = np.where(target == minor_class)[0]
    index_major_class = np.where(target == major_class)[0]

    # Calculate the oversampling size
    oversampling_size = len(index_major_class) - len(index_minor_class)
    
    if oversampling_size > 0:
        # Oversample the minority class
        new_minor = np.random.choice(index_minor_class,
                                     size=oversampling_size,
                                     replace=True)
        data = np.concatenate((data, data[new_minor]))
        target = np.concatenate((target, target[new_minor]))

    return data, target
