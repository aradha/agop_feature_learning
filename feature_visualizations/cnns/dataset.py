import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from loader import ImageNet


def get_imagenet(batch_size=128, path=None):
    if path is None:
        print("============================\n" + \
              "Please set path to ImageNet\n" + \
              "============================\n"
        )

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

    trainset = ImageNet(path, "train", transform)
    valset = ImageNet(path, "val", transform)

    trainloader = DataLoader(trainset,
                              batch_size=batch_size,
                              num_workers=1,
                              shuffle=True)

    valloader = DataLoader(valset,
                            batch_size=batch_size,
                            num_workers=1,
                            shuffle=False)

    print("Num Train: ", len(trainloader.dataset),
          "Num Test: ", len(valloader.dataset))

    return trainloader, valloader
