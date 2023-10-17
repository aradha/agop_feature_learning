import torch
import torch.nn as nn
import random
import numpy as np
from functorch import jacrev, vmap
from torch.nn.functional import pad
import dataset
from numpy.linalg import eig
from copy import deepcopy
from torch.linalg import norm
from torchvision import models

SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


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

class PatchConvLayer(nn.Module):
    def __init__(self, conv_layer):
        super().__init__()
        self.layer = conv_layer

    def forward(self, patches):
        out = torch.einsum('nwhcqr, kcqr -> nwhk', patches, self.layer.weight)
        n, w, h, k = out.shape
        out = out.transpose(1, 3).transpose(2, 3)
        return out

def get_jacobian(net, data, c_idx=0, chunk=100):
    with torch.no_grad():
        def single_net(x):
            return net(x.unsqueeze(0))[:,c_idx*chunk:(c_idx+1)*chunk].squeeze(0)
        return vmap(jacrev(single_net))(data)

def egop(model, z):
    ajop = 0
    c = 1000
    chunk_idxs = 100
    chunk = c // chunk_idxs
    for i in range(chunk_idxs):
        J = get_jacobian(model, z, c_idx=i, chunk=chunk)
        n, c, w, h, _, _, _ = J.shape
        J = J.transpose(1, 3).transpose(1, 2)
        grads = J.reshape(n*w*h, c, -1)
        ajop += torch.einsum('ncd, ncD -> dD', grads, grads)
    return ajop


def load_nn(net, init_net, layer_idx=0):

    count = 0
    for idx, m in enumerate(net.features):
        if isinstance(m, nn.Conv2d):
            count += 1
        if count-1 == layer_idx:
            l_idx = idx
            break

    patchnet = deepcopy(net)
    layer = PatchConvLayer(net.features[l_idx])

    (q, s) = net.features[l_idx].kernel_size
    (pad1, pad2) = net.features[l_idx].padding
    (s1, s2) = net.features[l_idx].stride

    patchnet.features = net.features[l_idx:]
    patchnet.features[0] = layer

    count = -1
    for idx, p in enumerate(net.parameters()):
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data
            _, ki, q, s = M.shape

            M = M.reshape(-1, ki*q*s)
            M = torch.einsum('nd, nD -> dD', M, M)

            M0 = [p for p in init_net.parameters()][idx]
            M0 = M0.reshape(-1, ki*q*s)
            M0 = torch.einsum('nd, nD -> dD', M0, M0)
            break

    return net, patchnet, M, M0, l_idx, [(q, s), (pad1,pad2), (s1,s2)]


def get_grads(net, patchnet, trainloader,
              kernel=(3,3), padding=(1,1),
              stride=(1,1), layer_idx=0):
    net.eval()
    net.cuda()
    patchnet.eval()
    patchnet.cuda()
    M = 0
    q, s = kernel
    pad1, pad2 = padding
    s1, s2 = stride

    # Num images for taking AGOP (Can be small for early layers)
    MAX_NUM_IMGS = 10

    for idx, batch in enumerate(trainloader):
        print("Computing GOP for sample " + str(idx) + \
              " out of " + str(MAX_NUM_IMGS))
        imgs, _ = batch
        with torch.no_grad():
            imgs = imgs.cuda()
            imgs = net.features[:layer_idx](imgs).cpu()
        patches = patchify(imgs, (q, s), (s1,s2), padding=(pad1,pad2))
        patches = patches.cuda()

        M += egop(patchnet, patches).cpu()
        del imgs, patches
        torch.cuda.empty_cache()
        if idx >= MAX_NUM_IMGS:
            break
    net.cpu()
    patchnet.cpu()
    return M


def min_max(M):
    return (M - M.min()) / (M.max() - M.min())


def correlation(M1, M2):
    M1 -= M1.mean()
    M2 -= M2.mean()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())

    return torch.sum(M1 * M2) / (norm1 * norm2)


def verify_NFA(net, init_net, trainloader, layer_idx=0):


    net, patchnet, M, M0, l_idx, conv_vals = load_nn(net,
                                                     init_net,
                                                     layer_idx=layer_idx)
    (q, s), (pad1, pad2), (s1, s2) = conv_vals

    i_val = correlation(M0, M)
    print("Correlation between Initial and Trained CNFM: ", i_val)

    G = get_grads(net, patchnet, trainloader,
                  kernel=(q, s),
                  padding=(pad1, pad2),
                  stride=(s1, s2),
                  layer_idx=l_idx)

    r_val = correlation(M, G)
    print("Correlation between Trained CNFM and AGOP: ", r_val)
    print("Final: ", i_val, r_val)

    return i_val.data.numpy(), r_val.data.numpy()


def main():

    # Adjust to index conv layers in VGGs
    idxs = list(range(8))

    fname = 'csv_logs/test.csv'
    outf = open(fname, 'w')

    net = models.vgg11(weights="DEFAULT")
    init_net = models.vgg11(weights=None)

    modules= list(net.children())[:-1]
    modules += [nn.Flatten(), list(net.children())[-1]]

    # Set path to imagenet data
    path = None

    trainloader, _ = dataset.get_imagenet(batch_size=2, path=path)

    for idx in idxs:
        i_val, r_val = verify_NFA(net, init_net, trainloader, layer_idx=idx)
        print("Layer " + str(idx+1) + ',' + str(i_val) + ',' + str(r_val), file=outf, flush=True)


if __name__ == "__main__":
    main()
