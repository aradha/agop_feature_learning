import torch
import torch.nn as nn
import random
import numpy as np
from functorch import jacrev, vmap
from torch.nn.functional import pad
import dataset
from numpy.linalg import eig
from copy import deepcopy
from torch.linalg import norm, svd
from torchvision import models
import torchvision


SEED = 2323

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)


def patchify(x, patch_size, stride_size, pad_type='zeros'):
    q1, q2 = patch_size
    s1, s2 = stride_size

    pad_1 = (q1-1)//2
    pad_2 = (q2-1)//2

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


class PatchBasicBlock(nn.Module):

    def __init__(self, block_layer, downsample=False):
        super().__init__()
        self.layer = block_layer
        self.downsample = downsample

    # x is patches instead of images
    def forward(self, X):
        x, y = X
        ops = self.layer
        _, _, _, _, q, s = y.shape
        z = y[:, :, :, :, (q-1)//2, (s-1)//2]
        z = z.transpose(1, 3).transpose(2, 3)

        s1, s2 = ops.conv1.layer.stride
        x = x[:, ::s1, ::s2, :, :, :]
        o = ops.conv1(x).contiguous()
        o = ops.bn1(o)
        o = ops.relu(o)
        o = ops.conv2(o)
        o = ops.bn2(o)

        if self.downsample:
            z = ops.downsample(z)
        o += z
        o = ops.relu(o)
        return o


class PatchBottleneck(nn.Module):

    def __init__(self, block_layer, downsample=False):
        super().__init__()
        self.layer = block_layer
        self.downsample = downsample

    # x is patches instead of images
    def forward(self, X):
        x, y = X
        ops = self.layer
        _, _, _, _, q, s = x.shape

        z = y[:, :, :, :, (q-1)//2, (s-1)//2]
        z = z.transpose(1, 3).transpose(2, 3)

        s1, s2 = ops.conv1.layer.stride
        x = x[:, ::s1, ::s2, :, :, :]
        o = ops.conv1(x).contiguous()
        o = ops.bn1(o)
        o = ops.relu(o)

        o = ops.conv2(o)
        o = ops.bn2(o)
        o = ops.relu(o)

        o = ops.conv3(o)
        o = ops.bn3(o)

        if self.downsample:
            z = ops.downsample(z)
        o += z
        o = ops.relu(o)
        return o


def get_jacobian(net, data, c_idx=0, chunk=100):
    count = 0
    with torch.no_grad():
        def single_net(x):
            o = net(x)[:,c_idx*chunk:(c_idx+1)*chunk]
            return o
        return vmap(jacrev(single_net))(data)


def egop(model, X):
    ajop = 0
    c = 1000
    chunk_idxs = 100
    chunk = c // chunk_idxs
    for i in range(chunk_idxs):
        J = get_jacobian(model, X, c_idx=i, chunk=chunk)[0]
        J = J[0, 0].transpose(0, 1)
        # n is number of images
        # c is number of channels
        # w, h give number of total patches
        n, c, w, h, _, _, _ = J.shape
        J = J.transpose(1, 3).transpose(1, 2)

        grads = J.reshape(n*w*h, c, -1)
        ajop += torch.einsum('ncd, ncD -> dD', grads, grads)
    return ajop


def load_nn(net, init_net,
            block_idx=0):

    patchnet = deepcopy(net)
    l_idx = block_idx
    layer_idx = 0
    subnet = net[:l_idx]
    for m in subnet.children():
        if isinstance(m, nn.Conv2d):
            layer_idx += 1
        # For Resnet18, 34
        elif isinstance(m, torchvision.models.resnet.BasicBlock):
        # For ResNet50 -> 152
        #elif isinstance(m, torchvision.models.resnet.Bottleneck):
            modules = [mod for mod in m.modules() if not isinstance(mod, nn.Sequential)]
            for mod in modules:
                if isinstance(mod, nn.Conv2d):
                    layer_idx += 1

    patchnet = patchnet[l_idx:]
    if block_idx == 0:
        patchnet[0] = PatchConvLayer(patchnet[0])
    else:

        if patchnet[0].downsample is not None:
            downsample = True
        else:
            downsample = False

        # For Resnet18, 34
        patchnet[0].conv1 = PatchConvLayer(patchnet[0].conv1)
        patchnet[0] = PatchBasicBlock(patchnet[0], downsample=downsample)

        # For Resnet50 -> 152
        #patchnet[0].conv1 = PatchConvLayer(patchnet[0].conv1)
        #patchnet[0] = PatchBottleneck(patchnet[0], downsample=downsample)

    count = -1
    for idx, p in enumerate(net.parameters()):
        if len(p.shape) > 1:
            count += 1
        if count == layer_idx:
            M = p.data
            _, ki, q, s = M.shape

            M0 = [p for p in init_net.parameters()][idx].data

            M = M.reshape(-1, ki*q*s)
            M = torch.einsum('nd, nD -> dD', M, M)

            M0 = M0.reshape(-1, ki*q*s)
            M0 = torch.einsum('nd, nD -> dD', M0, M0)
            break

    return net, patchnet, M, M0, l_idx, (q, s)


def get_grads(net, patchnet, trainloader,
              kernel=(3,3),
              layer_idx=0):
    net.eval()
    net.cuda()
    patchnet.eval()
    patchnet.cuda()
    M = 0
    q,s = kernel

    # Num images for taking AGOP (set to >100 for deeper layers)
    MAX_NUM_IMGS = 10
    for idx, batch in enumerate(trainloader):
        print("Computing GOP for sample " + str(idx) + \
              " out of " + str(MAX_NUM_IMGS))
        imgs, _ = batch
        with torch.no_grad():
            imgs = imgs.cuda()
            imgs = net[:layer_idx](imgs).cpu()
        patches = patchify(imgs, (q, s), (1, 1))
        p_copy = deepcopy(patches)
        patches = patches.cuda()
        p_copy = p_copy.cuda()

        M += egop(patchnet, [patches.unsqueeze(0), p_copy.unsqueeze(0)]).cpu()
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
    # To avoid numerical issues with small values
    M1 /= M1.max()
    M2 /= M2.max()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())
    return torch.sum(M1 * M2) / (norm1 * norm2)


def verify_NFA(net, init_net, trainloader, layer_idx=0):

    net, patchnet, M, M0, l_idx, (q, s) = load_nn(net,
                                                  init_net,
                                                  block_idx=layer_idx)

    i_val = correlation(M0, M)
    print("Correlation between Initial and Trained CNFM: ", i_val)

    G = get_grads(net, patchnet, trainloader, kernel=(q, s), layer_idx=l_idx)
    G = sqrt(G)

    r_val = correlation(M, G)

    print("Correlation between Trained CNFM and AGOP: ", r_val)
    print("Final: ", i_val, r_val)

    return i_val.data.numpy(), r_val.data.numpy()


def subroutine_unroll_net(net):
    modules = list(net.children())
    unrolled = []
    for m in modules:
        if isinstance(m, nn.Sequential):
            unrolled += subroutine_unroll_net(m)
        else:
            unrolled.append(m)
    return unrolled


def unroll_net(net):
    modules = subroutine_unroll_net(net)[:-1]
    modules += [nn.Flatten(), list(net.children())[-1]]
    net = nn.Sequential(*modules)
    return net


def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def main():

    # Set indices for BasicBlocks or Bottlenecks
    # Note: Do not start at 0-3 since first layers are not part of a
    #       block or bottleneck
    idxs = list(range(4, 12))

    fname = 'csv_logs/test.csv'
    outf = open(fname, 'w')

    net = models.resnet18(weights="DEFAULT")
    init_net = models.resnet18(weights=None)

    net = unroll_net(net)
    init_net = unroll_net(init_net)

    # Set path to imagenet data
    path = None

    # Batch size should be 1 to avoid issues with grads for skip connections
    trainloader, _ = dataset.get_imagenet(batch_size=1, path=path)

    for idx in idxs:
        i_val, r_val = verify_NFA(net, init_net, trainloader, layer_idx=idx)
        print("Layer " + str(idx+1) + ',' + str(i_val) + ',' + str(r_val), file=outf, flush=True)


if __name__ == "__main__":
    main()
