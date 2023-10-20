import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import random
import os

from math import log, sqrt

def one_hot_data(dataset, num_classes, num_samples):
    Xs = []
    ys = []

    for ix in range(min(len(dataset),num_samples)):
        X,y = dataset[ix]
        Xs.append(X)
        
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        ys.append(ohe_y)

    return torch.stack(Xs), torch.stack(ys)

def get_binary(dataset, classes):
    c1, c2 = classes
    
    binary_dataset = []
    for ix in tqdm(range(len(dataset))):
        X,y = dataset[ix]
        
        if y==c1:
            binary_dataset.append((X,0))
        elif y==c2:
            binary_dataset.append((X,1))

    return binary_dataset

def get_toy(n_train, n_test, sigma):
    d = 32
    k = 16
    
    wstar = torch.ones((d,))
    wstar /= torch.linalg.norm(wstar)
    
    path = os.environ["DATA_PATH"]
    train_X = torch.load(os.path.join(path, f'toy/train_frames_n_{n_train}.pt'))
    train_y = torch.load(os.path.join(path, f'toy/train_y_n_{n_train}.pt'))
    test_X = torch.load(os.path.join(path, f'toy/test_frames_n_{n_test}.pt'))
    test_y = torch.load(os.path.join(path, f'toy/test_y_n_{n_test}.pt'))
    wstar_train = torch.load(os.path.join(path, f'toy/train_wstar_n_{n_train}.pt'))
    wstar_test = torch.load(os.path.join(path, f'toy/test_wstar_n_{n_test}.pt'))
    
    std = sqrt(sigma)/(sqrt(d)) 

    train_X = train_X*std
    test_X = torch.randn(size=(n_test, 1, k, d))*std

    for i in range(n_train):
        wstar_idx = wstar_train[i]
        y = train_y[i]
        train_X[i,0,wstar_idx] = wstar*y

    for i in range(n_test):
        wstar_idx = wstar_test[i]
        y = test_y[i]
        test_X[i,0,wstar_idx] = wstar*y

    ohe_train_y = torch.zeros(n_train, 2)
    ohe_train_y[train_y==1,0] = 1
    ohe_train_y[train_y==-1,1] = 1

    ohe_test_y = torch.zeros(n_test, 2)
    ohe_test_y[test_y==1,0] = 1
    ohe_test_y[test_y==-1,1] = 1
    
    return train_X, test_X, ohe_train_y, ohe_test_y

def get_toy_mnist(n_train, n_test, sigma):

    path = os.environ["DATA_PATH"]

    train_X = torch.load(os.path.join(path, f'toy_mnist/train_X_n_{n_train}.pt'))
    train_y = torch.load(os.path.join(path, f'toy_mnist/train_y_n_{n_train}.pt'))
    train_frame = torch.load(os.path.join(path, f'toy_mnist/train_frames_n_{n_train}.pt'))
    train_s1s = torch.load(os.path.join(path, f'toy_mnist/train_frame_s1s_n_{n_train}.pt'))
    train_s2s = torch.load(os.path.join(path, f'toy_mnist/train_frame_s2s_n_{n_train}.pt'))

    test_X = torch.load(os.path.join(path, f'toy_mnist/test_X_n_{n_test}.pt'))
    test_y = torch.load(os.path.join(path, f'toy_mnist/test_y_n_{n_test}.pt'))
    test_frame = torch.load(os.path.join(path, f'toy_mnist/test_frames_n_{n_test}.pt'))
    test_s1s = torch.load(os.path.join(path, f'toy_mnist/test_frame_s1s_n_{n_test}.pt'))
    test_s2s = torch.load(os.path.join(path, f'toy_mnist/test_frame_s2s_n_{n_test}.pt'))

    def noise_frame(X, frame, s1s, s2s, std):
        n, c, p, q = X.shape
        P, Q = 42, 42
        
        e1s = s1s + p
        e2s = s2s + q

        framed_X = frame*std

        for i in range(n):
            s1 = s1s[i]
            s2 = s2s[i]
            e1 = e1s[i]
            e2 = e2s[i]
            framed_X[i,0,s1:e1,s2:e2] = X[i,0]

        return framed_X

    train_X = noise_frame(train_X, train_frame, train_s1s, train_s2s, std=0.5*sigma**0.5)
    test_X = noise_frame(test_X, test_frame, test_s1s, test_s2s, std=0.5*sigma**0.5)

    return train_X, test_X, train_y, test_y

def get_cifar(n_train, n_test):

    NUM_CLASSES = 10
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "cifar10/"

    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,
                                            download=False)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.CIFAR10(root=path,
                                           train=False,
                                           transform=transform,
                                           download=False)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_cifar100(n_train, n_test):

    NUM_CLASSES = 100
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "cifar100/"

    trainset = torchvision.datasets.CIFAR100(root=path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.CIFAR100(root=path,
                                           train=False,
                                           transform=transform,
                                           download=True)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_svhn(n_train, n_test):

    NUM_CLASSES = 10

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + 'SVHN/'

    trainset = torchvision.datasets.SVHN(root=path,
                                         split='train',
                                         transform=transform,
                                         download=False)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.SVHN(root=path,
                                        split='test',
                                        transform=transform,
                                        download=False)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_flowers(class1=-1, class2=-1, get_all=False):
    
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'Flowers/'
    trainset = torchvision.datasets.Flowers102(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)

    testset = torchvision.datasets.Flowers102(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    
    if get_all:
        NUM_CLASSES=102
        train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=10000)
        test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=10000)
    else:
        trainset = get_binary(trainset, classes=(class1, class2))
        testset = get_binary(testset, classes=(class1, class2))

        NUM_CLASSES = 2
        train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=1000)
        test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=1000)

    return train_X, test_X, train_y, test_y

def get_dtd(class1=-1, class2=-1, get_all=False):
    
    transform = transforms.Compose(
        [
            transforms.Resize(size=(32,32)),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'dtd/'
    trainset = torchvision.datasets.DTD(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)

    testset = torchvision.datasets.DTD(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    if get_all:
        NUM_CLASSES=47
        train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=10000)
        test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=10000)
    else:
        trainset = get_binary(trainset, classes=(class1, class2))
        testset = get_binary(testset, classes=(class1, class2))

        NUM_CLASSES = 2
        train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=1000)
        test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=1000)

    return train_X, test_X, train_y, test_y


def get_FGVCAircraft(n_train, n_test):

    NUM_CLASSES = 30
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'FGVCAircraft/'
    trainset = torchvision.datasets.FGVCAircraft(root=data_path,
                                        split  = "train",
                                        annotation_level='manufacturer',
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.FGVCAircraft(root=data_path,
                                        split = "test",
                                        annotation_level='manufacturer',
                                        transform=transform,
                                        download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_food(n_train=1000, n_test=1000):
    transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
    ])

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'Food101/'

    trainset = torchvision.datasets.Food101(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.Food101(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    
    NUM_CLASSES=101

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y


def get_PCAM(n_train, n_test):

    NUM_CLASSES = 2
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'PCAM/'
    trainset = torchvision.datasets.PCAM(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.PCAM(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y


def get_StanfordCars(n_train, n_test):

    transform = transforms.Compose([
        transforms.Resize(size=(32,32)),
        transforms.ToTensor()
    ])

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'StanfordCars/'
    trainset = torchvision.datasets.StanfordCars(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.StanfordCars(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    
    NUM_CLASSES = 196
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_FashionMNIST(n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'FashionMNIST/'
    trainset = torchvision.datasets.FashionMNIST(root=data_path,
                                        train=True,
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.FashionMNIST(root=data_path,
                                        train=False,
                                        transform=transform,
                                        download=True)

    NUM_CLASSES = 10

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_QMNIST(n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'QMNIST/'
    trainset = torchvision.datasets.QMNIST(root=data_path,
                                        what='train',
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.QMNIST(root=data_path,
                                        what='test',
                                        transform=transform,
                                        download=True)

    NUM_CLASSES = 10

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_EMNIST_binary(class1, class2):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'EMNIST/'
    trainset = torchvision.datasets.EMNIST(root=data_path,
                                        split="letters",
                                        transform=transform,
                                        download=True)
    n_tot = len(trainset)
    n_train_sub = int(0.8*n_tot)
    n_test_sub = n_tot-n_train_sub
    trainset, testset = torch.utils.data.random_split(trainset, [n_train_sub, n_test_sub])
    
    trainset = get_binary(trainset, classes=(class1, class2))
    testset = get_binary(testset, classes=(class1, class2))

    NUM_CLASSES = 2
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=50)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=50)

    return train_X, test_X, train_y, test_y

def get_EMNIST(n_train, n_test):

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'EMNIST/'
    trainset = torchvision.datasets.EMNIST(root=data_path,
                                        split="letters",
                                        transform=transform,
                                        download=True)
    n_tot = len(trainset)
    n_train_sub = int(0.8*n_tot)
    n_test_sub = n_tot-n_train_sub
    trainset, testset = torch.utils.data.random_split(trainset, [n_train_sub, n_test_sub])
    
    NUM_CLASSES = 27
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_Caltech101(n_train, n_test):

    transform = transforms.Compose(
        [
            transforms.Resize(size=(32,32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'Caltech101/'
    trainset = torchvision.datasets.Caltech101(root=data_path,
                                        transform=transform,
                                        download=True)

    n_tot = len(trainset)
    n_train = int(0.8*n_tot)
    n_test = n_tot-n_train
    trainset, testset = torch.utils.data.random_split(trainset, [n_train, n_test])

    NUM_CLASSES = 101

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_celeba(class1, class2):

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'CelebA/'
    trainset = torchvision.datasets.CelebA(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.CelebA(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = get_binary(trainset, classes=(class1, class2))
    testset = get_binary(testset, classes=(class1, class2))

    NUM_CLASSES = 2
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=10000)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=10000)

    return train_X, test_X, train_y, test_y

def get_gtsrb_binary(class1, class2):

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'GTSRB/'
    trainset = torchvision.datasets.GTSRB(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.GTSRB(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = get_binary(trainset, classes=(class1, class2))
    testset = get_binary(testset, classes=(class1, class2))

    NUM_CLASSES = 2
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=10000)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=10000)

    return train_X, test_X, train_y, test_y

def get_gtsrb(n_train, n_test):

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'GTSRB/'
    trainset = torchvision.datasets.GTSRB(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.GTSRB(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    NUM_CLASSES = 43

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_stl10_binary(class1, class2):

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'STL10/'
    trainset = torchvision.datasets.STL10(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.STL10(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)

    trainset = get_binary(trainset, classes=(class1, class2))
    testset = get_binary(testset, classes=(class1, class2))

    NUM_CLASSES = 2
    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=500)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=500)

    return train_X, test_X, train_y, test_y

def get_stl10(n_train, n_test):

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'STL10/'
    trainset = torchvision.datasets.STL10(root=data_path,
                                        split  = "train",
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.STL10(root=data_path,
                                        split = "test",
                                        transform=transform,
                                        download=True)
    NUM_CLASSES = 10

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y

def get_USPS(n_train, n_test):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data_path = os.environ['DATA_PATH']
    data_path = data_path + 'USPS/'
    trainset = torchvision.datasets.USPS(root=data_path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.USPS(root=data_path,
                                        train = False,
                                        transform=transform,
                                        download=True)


    NUM_CLASSES = 10

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    return train_X, test_X, train_y, test_y
