"""This module contains utility functions for the project."""
import os
import random
import logging
from typing import Iterable, List
from pydvl.value.loo import naive_loo
from pydvl.value.shapley import compute_shapley_values
from pydvl.utils import Utility
from pydvl.value.shapley.truncated import RelativeTruncation
from pydvl.value import MaxUpdates, HistoryDeviation, compute_banzhaf_semivalues, compute_beta_shapley_semivalues
from pydvl.utils.dataset import Dataset
from pydvl.value.result import ValuationResult
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd

from itertools import chain, combinations
from sklearn.linear_model import LogisticRegression

__all__ = [
    "set_random_seed",
    "setup_logger",
    "equilibrate_clases",
    "make_balance_sample",
    "convert_values_to_dataframe",
]

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logger() -> logging.Logger:
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

def undersamp_equilibration(
        data: np.ndarray,
        target: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Equilibrate the classes of a dataset with
    two classes in the target variable,
    using undersampling.

    Args:
        df (pd.DataFrame): The dataset.

    Returns:
        pd.DataFrame: The balanced dataset.
    """
    unique_targets, counts = np.unique(target, return_counts=True)
    min_count = counts.min()
    
    balanced_indices = np.concatenate([
        np.random.choice(np.where(target == class_label)[0], min_count, replace=False)
        for class_label in unique_targets
    ])
    
    # Use the indices to extract the corresponding rows from data and target
    balanced_data = data[balanced_indices]
    balanced_target = target[balanced_indices]
    
    return balanced_data, balanced_target

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

def compute_values(
    method_name: str,
    utility: Utility
) -> ValuationResult:
    """
    Compute the data values for a given method and utility.

    Args:
        method_name (str): The name of the method to use.
        utility (Utility): The utility to use. (pyDVL)

    Returns:
        ValuationResult: The values. (pyDVL)
    """
    if method_name == "LOO":
        values = naive_loo(utility, progress=False)
    elif method_name == "Shapley":
        values = compute_shapley_values(
            u=utility,
            mode="permutation_montecarlo",
            done=MaxUpdates(500) | HistoryDeviation(n_steps=100, rtol=0.05),
            truncation=RelativeTruncation(utility, rtol=0.01)
        )
    elif method_name == "Banzhaf":
        values = compute_banzhaf_semivalues(
            u=utility,
            done=MaxUpdates(500)
        )
    elif method_name == "Beta-1-16":
        values = compute_beta_shapley_semivalues(
            u=utility,
            alpha=1,
            beta=16,
            done=MaxUpdates(500)
        )
    elif method_name == "Beta-1-4":
        values = compute_beta_shapley_semivalues(
            u=utility,
            alpha=1,
            beta=4,
            done=MaxUpdates(500)
        )
    elif method_name == "Beta-16-1":
        values = compute_beta_shapley_semivalues(
            u=utility,
            alpha=16,
            beta=1,
            done=MaxUpdates(500)
        )
    elif method_name == "Beta-4-1":
        values = compute_beta_shapley_semivalues(
            u=utility,
            alpha=4,
            beta=1,
            done=MaxUpdates(500)
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return values

def convert_values_to_dataframe(values: ValuationResult) -> pd.DataFrame:
    """
    Convert the values of a ValuationResult object to a DataFrame.

    Args:
        values (ValuationResult): The values to convert.

    Returns:
        pd.DataFrame: The DataFrame with the values.
    """
    df = (
        values.to_dataframe(column="value")
        .drop(columns=["value_stderr"])
        .T.reset_index(drop=True)
    )
    df = df[sorted(df.columns)]
    return df

# def f1_misslabel(data_values: pd.DataFrame) -> float:
#     """
#     Computes the F1 score for a prediction based on
#     a threshold derived from the input data.

#     Args:
#         data_values (pd.DataFrame): The data values.

#     Returns:
#         float: The f1 score.
#     """
#     # Get the number of data points
#     # and initialize the arrays
#     n_data = len(data_values)
#     pred = np.zeros(n_data)
#     true = np.zeros(n_data)

#     # Extract the values from the "value" column and
#     # compute the threshold
#     value_column = data_values['value'].values
#     threshold = np.sort(value_column)[int(0.1 * n_data)]
    
#     pred[value_column < threshold] = 1
#     true[:int(0.1 * n_data)] = 1
#     return f1_score(true, pred)

def f1_misslabel(value_array: np.ndarray) -> float:
    """
    Computes the F1 score for a prediction based on
    a threshold derived from the input data.

    Args:
        value_array (np.ndarray): The data values.

    Returns:
        float: The f1 score.
    """
    # Get the number of data points
    # and initialize the arrays
    n_data = len(value_array)
    pred = np.zeros(n_data)
    true = np.zeros(n_data)

    # Compute the threshold
    threshold = np.sort(value_array)[int(0.1 * n_data)]
    
    pred[value_array < threshold] = 1
    true[:int(0.1 * n_data)] = 1
    
    # Compute and return the F1 score
    return float(f1_score(true, pred))



