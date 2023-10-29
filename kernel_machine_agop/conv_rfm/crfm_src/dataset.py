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

import star_dataset

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


def get_toy_mnist(n_train, n_test, sigma):

    NUM_CLASSES = 10
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "MNIST/"
    trainset = torchvision.datasets.MNIST(root=path,
                                        train = True,
                                        transform=transform,
                                        download=True)
    testset = torchvision.datasets.MNIST(root=path,
                                        train = False,
                                        transform=transform,
                                        download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)
    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    def get_noise_frame(X):
        n, c, p, q = X.shape
        P, Q = 42, 42

        s1s = torch.randint(high=(P-p), size=(n,))
        s2s = torch.randint(high=(Q-q), size=(n,))

        X_frame = torch.randn(size=(n,c,P,Q))

        return X_frame, s1s, s2s
    
    train_frame, train_s1s, train_s2s = get_noise_frame(train_X)
    test_frame, test_s1s, test_s2s = get_noise_frame(test_X)

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

def get_star(n_train, n_test):

    X, y = star_dataset.star_dataset(n_train + n_test)

    train_X = X[:n_train]
    test_X = X[n_train:]

    train_y = y[:n_train]
    test_y = y[n_train:]

    return train_X, test_X, train_y, test_y
                                            