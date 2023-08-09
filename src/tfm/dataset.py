from pydvl.utils.dataset import Dataset
from utils import build_pyDVL_dataset

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
        root='.',
        train=True,
        download=True
    )
    testset = datasets.MNIST(
        root='.',
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
        root='.',
        train=True,
        download=True
    )
    testset = datasets.FashionMNIST(
        root='.',
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


# TODO: El tipo de retorno del CIFAR10?
def create_cifar() -> Dataset:
    """
    Create the CIFAR10 dataset.

    Returns:
        Dataset (pyDVL.Dataset): The CIFAR10 dataset.
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


def make_balance_sample(data, target):
    p = np.mean(target)
    if p < 0.5:
        minor_class=1
    else:
        minor_class=0
    
    index_minor_class = np.where(target == minor_class)[0]
    n_minor_class=len(index_minor_class)
    n_major_class=len(target)-n_minor_class
    new_minor=np.random.choice(index_minor_class, size=n_major_class-n_minor_class, replace=True)

    data=np.concatenate([data, data[new_minor]])
    target=np.concatenate([target, target[new_minor]])
    return data, target

def get_minidata(dataset):
    """Función del chino pero mejorada, hay que ver el tema de lo
    que retorna"""

    # TODO: Esta direccion está mal
    open_ml_path = 'OpenML_datasets/'

    np.random.seed(999)

    # Dictionary to map dataset names to their respective file names
    dataset_mapping = {
        'fraud': 'CreditCardFraudDetection_42397.pkl',
        'apsfail': 'APSFailure_41138.pkl',
        'click': 'Click_prediction_small_1218.pkl',
        'phoneme': 'phoneme_1489.pkl',
        'wind': 'wind_847.pkl',
        'pol': 'pol_722.pkl',
        'creditcard': 'default-of-credit-card-clients_42477.pkl',
        'cpu': 'cpu_act_761.pkl',
        'vehicle': 'vehicle_sensIT_357.pkl',
        '2dplanes': '2dplanes_727.pkl'
    }

    if dataset in dataset_mapping:
        data_dict = pickle.load(open(open_ml_path + dataset_mapping[dataset], 'rb'))
        data, target = data_dict['X_num'], data_dict['y']
        target = (target == 1).astype(np.int32)
        data, target = make_balance_sample(data, target)

        idxs = np.random.permutation(len(data))
        data, target = data[idxs], target[idxs]
        return data, target, None, None 
    else:
        print('No such dataset!')
        sys.exit(1)
