from pydvl.utils.dataset import Dataset

import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms

import numpy as np
import sys
import pickle

def create_mnist() -> Dataset:
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.MNIST(
        root='.',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = datasets.MNIST(
        root='.',
        train=False,
        download=True,
        transform=transform_test
    )

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    mnist = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    return mnist

def create_fmnist() -> Dataset:
    transform_train = transforms.Compose([transforms.ToTensor(),])
    transform_test = transforms.Compose([transforms.ToTensor(),])
    trainset = datasets.FashionMNIST(
        root='.',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = datasets.FashionMNIST(
        root='.',
        train=False,
        download=True,
        transform=transform_test
    )

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    # Redundant
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()

    fmnist = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    return fmnist


def create_cifar() -> Dataset:
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root='.',
        train=True,
        download=True,
        transform=transform_train
    ) 
    testset = torchvision.datasets.CIFAR10(
        root='.',
        train=False,
        download=True,
        transform=transform_test
    )

    (x_train, y_train), (x_test, y_test) = (trainset.data, trainset.targets), (testset.data, testset.targets)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    cifar = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    return cifar


def create_dogcat():

    x_train, y_train, x_test, y_test = create_cifar()

    dogcat_ind = np.where(np.logical_or(y_train==3, y_train==5))[0]
    x_train, y_train = x_train[dogcat_ind], y_train[dogcat_ind]
    y_train[y_train==3] = 0
    y_train[y_train==5] = 1

    dogcat_ind = np.where(np.logical_or(y_test==3, y_test==5))[0]
    x_test, y_test = x_test[dogcat_ind], y_test[dogcat_ind]
    y_test[y_test==3] = 0
    y_test[y_test==5] = 1

    return x_train, y_train, x_test, y_test


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

