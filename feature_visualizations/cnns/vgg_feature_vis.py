import torch
import torch.nn as nn
import random
import numpy as np
from functorch import jacrev, vmap
from torch.nn.functional import pad
import dataset
import visdom
from numpy.linalg import eig
from copy import deepcopy
from torch.linalg import norm
from torchvision import models
import hickle
from torch.nn.functional import fold
from torch.linalg import norm, eig


SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

vis = visdom.Visdom('http://127.0.0.1', use_incoming_socket=False)
vis.close(env='main')

def patchify(x, patch_size, stride_size, padding=None, pad_type='zeros'):
    q1, q2 = patch_size
    s1, s2 = stride_size

    if padding is None:
        pad_1 = (q1-1)//2
        pad_2 = (q2-1)//2
    else:
        pad_1, pad_2 = padding

    pad_dims = (pad_2, pad_2, pad_1, pad_1)
    if pad_type == 'zeros':
        x = pad(x, pad_dims)
    elif pad_type == 'circular':
        x = pad(x, pad_dims, 'circular')

    patches = x.unfold(2, q1, s1).unfold(3, q2, s2)
    patches = patches.transpose(1, 3).transpose(1, 2)
    return patches

def min_max(M):
    return (M - M.min()) / (M.max() - M.min())


def transform_image(net, img, G, layer_idx=0):

    count = -1
    for idx, p in enumerate(net.parameters()):
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data
            print(M.shape)
            _, ki, q, s = M.shape

            M = M.reshape(-1, ki*q*s)
            M = torch.einsum('nd, nD -> dD', M, M)
            break

    count = 0
    l_idx = None
    for idx, m in enumerate(net.features):
        if isinstance(m, nn.Conv2d):
            print(m, count)
            count += 1

        if count-1 == layer_idx:
            l_idx = idx
            break

    net.eval()
    net.cuda()
    img = img.cuda()
    img = net.features[:l_idx](img).cpu()
    net.cpu()

    if G is not None:
        M = G

    patches = patchify(img, (q, s), (1, 1))

    n, w, h, q, s, c = patches.shape
    patches = patches.reshape(n, w, h, q*s*c)
    M_patch = torch.einsum('nwhd, dD -> nwhD', patches, M)
    M_patch = norm(M_patch, dim=-1)

    vis.image(min_max(M_patch[0]))


def main():

    net = models.vgg19(pretrained=True)

    modules= list(net.children())[:-1]
    modules += [nn.Flatten(), list(net.children())[-1]]

    layer_idx = 0

    G = None # PATH TO VGG19 AGOP for layer given by layer_idx
    IMAGENET_PATH = None # PATH TO IMAGENET

    trainloader, testloader = dataset.get_imagenet(batch_size=2, path=IMAGENET_PATH)

    for idx, batch in enumerate(testloader):
        imgs, labels = batch
        vis.image(min_max(imgs[0]))
        vis.image(min_max(imgs[1]))
        transform_image(net, imgs[:1], G, layer_idx=layer_idx)
        transform_image(net, imgs[1:2], G, layer_idx=layer_idx)
        if idx == 40:
            break


if __name__ == "__main__":
    main()
