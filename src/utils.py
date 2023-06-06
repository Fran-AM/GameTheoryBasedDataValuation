"""This module contains utility functions for the project."""
import os
import random

import torch

import numpy as np
import pandas as pd

# TODO: Set seed for cuda
def set_random_seed(seed:int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

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
