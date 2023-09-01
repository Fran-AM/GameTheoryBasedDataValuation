from pydvl.utils.dataset import Dataset
from utils.utils import build_pyDVL_dataset, undersamp_equilibration

import torchvision.datasets as datasets
import torchvision

import numpy as np
import sys
import pickle

def create_mnist() -> Dataset:
    """
    Create the MNIST dataset.

    Returns:
        Dataset (pyDVL.Dataset): The MNIST dataset.
    """
    trainset = datasets.MNIST(
        root='../../data',
        train=True,
        download=True
    )
    testset = datasets.MNIST(
        root='../../data',
        train=False,
        download=True
    )

    # Nota para el reshaping, estas imagenes son de 28x28
    return build_pyDVL_dataset(
        X_train=trainset.data,
        y_train=trainset.targets,
        X_test=testset.data,
        y_test=testset.targets
    )

def create_fmnist() -> Dataset:
    """
    Create the Fashion MNIST dataset.

    Returns:
        Dataset (pyDVL.Dataset): The Fashion MNIST dataset.
    """
    trainset = datasets.FashionMNIST(
        root='../../data',
        train=True,
        download=True
    )
    testset = datasets.FashionMNIST(
        root='../../data',
        train=False,
        download=True
    )

    # Nota para el reshaping, estas imagenes son de 28x28
    return build_pyDVL_dataset(
        X_train=trainset.data,
        y_train=trainset.targets,
        X_test=testset.data,
        y_test=testset.targets
    )

def create_cifar() -> Dataset:
    """
    Create the CIFAR10 dataset.

    Returns:
        Dataset (pyDVL.Dataset): The CIFAR10 dataset.
    """
    trainset = torchvision.datasets.CIFAR10(
        root='../../data',
        train=True,
        download=True
    ) 
    testset = torchvision.datasets.CIFAR10(
        root='../../data',
        train=False,
        download=True
    )
    # Nota para el reshaping, estas imagenes son de 32x32x3
    return build_pyDVL_dataset(
        X_train=trainset.data,
        y_train=trainset.targets,
        X_test=testset,
        y_test=testset
    )

def create_dogcat() -> Dataset:
    """
    Extract dog and cat images from CIFAR10 dataset.

    Returns:
        Dataset (pyDVL.Dataset): The CIFAR10 dataset with
        only dog and cat images.
    """

    """
    Warning, esta funcion no ha sido testeada.
    """
    trainset = torchvision.datasets.CIFAR10(
        root='.',
        train=True,
        download=True
    ) 
    testset = torchvision.datasets.CIFAR10(
        root='.',
        train=False,
        download=True
    )

    x_train = np.array(trainset.data)/255.0
    x_test = np.array(testset.data)/255.0
    y_train = np.array(trainset.targets)
    y_test = np.array(testset.targets)

    dogcat_ind = np.where(np.logical_or(y_train==3, y_train==5))[0]
    x_train, y_train = x_train[dogcat_ind], y_train[dogcat_ind]
    y_train[y_train==3] = 0
    y_train[y_train==5] = 1

    dogcat_ind = np.where(np.logical_or(y_test==3, y_test==5))[0]
    x_test, y_test = x_test[dogcat_ind], y_test[dogcat_ind]
    y_test[y_test==3] = 0
    y_test[y_test==5] = 1

    return build_pyDVL_dataset(
        X_train=x_train,
        y_train=trainset.targets,
        X_test=testset,
        y_test=testset
    )

def get_openML_data(
        dataset: str,
        n_data: int,
        n_test: int,
        flip_ratio: float = 0.0,
    )->Dataset:
    """
    Read and preprocess the datasets from OpenML.

    Args:
        dataset (str): The name of the dataset.
        n_data (int): The number of data points to use.
        n_test (int): The number of test points to use.
        flip_ratio (float): The ratio of flipped labels.

    Returns:
        Dataset (pyDVL.Dataset): The dataset.
    """

    openML_path = '../../data/openML/'

    # Dictionary to map dataset names to their respective file names
    ds_map = {
        'apsfail': 'APSFailure_41138.pkl',
        'click': 'Click_prediction_small_1218.pkl',
        'phoneme': 'phoneme_1489.pkl',
        'wind': 'wind_847.pkl',
        'pol': 'pol_722.pkl',
        'cpu': 'cpu_act_761.pkl',
        '2dplanes': '2dplanes_727.pkl'
    }

    if dataset in ds_map:
        try:
            data_dict = pickle.load(open(openML_path + ds_map[dataset], 'rb'))
            data, target = data_dict['X_num'], data_dict['y']
            target = (target == 1).astype(np.int32)
            data, target = undersamp_equilibration(data, target)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found for dataset '{dataset}'")
    else:
        raise ValueError(f"Dataset '{dataset}' not found")
    
    x_train, y_train = data[:n_data], target[:n_data]
    x_test, y_test = data[n_data:n_data+n_test],target[n_data:n_data+n_test]

    # Normalization
    x_mean, x_std= np.mean(x_train, 0), np.std(x_train, 0)
    norm = lambda x: (x - x_mean) / np.clip(x_std, 1e-12, None)
    x_train, x_test = norm(x_train), norm(x_test)

    # Flip labels
    if len(y_train.shape) != 1:
        raise ValueError("Expected y_train to be a 1-dimensionalarray, "
                         "but got a different shape.")

    n_flip = int(n_data*flip_ratio)
    y_train[:n_flip] = 1 - y_train[:n_flip]

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

