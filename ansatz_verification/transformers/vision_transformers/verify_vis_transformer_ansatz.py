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
import math
import torch.nn.functional as F
from torch.nn.functional import fold

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

HEAD_IDX = None
HEAD_IDXS = None
PERMUTATION = None

def set_head_idx(head_idx, num_heads=12):
    global HEAD_IDX, HEAD_IDXS, PERMUTATION
    HEAD_IDX = head_idx
    HEAD_IDXS = list((i for i in range(num_heads) if i != HEAD_IDX))
    PERMUTATION = list([i+1 for i in range(num_heads-1)])
    PERMUTATION.insert(HEAD_IDX, 0)
    PERMUTATION = torch.LongTensor(PERMUTATION).cuda()

def min_max(M):
    return (M - M.min()) / (M.max() - M.min())

def correlation(M1, M2):
    M1 -= M1.mean()
    M2 -= M2.mean()

    # For numerical stability
    M1 /= M1.max()
    M2 /= M2.max()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())
    return torch.sum(M1 * M2) / (norm1 * norm2)


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


class CustomMHA(nn.Module):

    def __init__(self, attn):
        super().__init__()

        self.n_embd = attn.embed_dim
        self.n_head = attn.num_heads
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, bias=True)

        for idx, param in enumerate(self.c_attn.parameters()):
            if idx == 0:
                param.data = attn.in_proj_weight
            else:
                param.data = attn.in_proj_bias
        self.c_proj = attn.out_proj
        self.dropout = attn.dropout

    def forward(self, X):

        x1, x2, x3, x = X  # x is generic input we will ignore when we take grads

        B, T, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, _, _  = self.c_attn(x1).split(self.n_embd, dim=2)
        _, k, _  = self.c_attn(x2).split(self.n_embd, dim=2)
        _, _, v  = self.c_attn(x3).split(self.n_embd, dim=2)

        qt, kt, vt = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        kt = kt.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        qt = qt.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        vt = vt.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        kf = torch.cat([k[:, HEAD_IDX:HEAD_IDX+1, :, :], kt[:, HEAD_IDXS, :, :]], dim=1)
        qf = torch.cat([q[:, HEAD_IDX:HEAD_IDX+1, :, :], qt[:, HEAD_IDXS, :, :]], dim=1)
        vf = torch.cat([v[:, HEAD_IDX:HEAD_IDX+1, :, :], vt[:, HEAD_IDXS, :, :]], dim=1)

        kf = kf.index_select(1, PERMUTATION)
        qf = qf.index_select(1, PERMUTATION)
        vf = vf.index_select(1, PERMUTATION)

        att = (qf @ kf.transpose(-2, -1)) * (1.0 / math.sqrt(kt.size(-1)))
        att = F.softmax(att, dim=-1)

        y = att @ vf # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class CustomBlock(torch.nn.Module):

    def __init__(self, block_layer):
        super(CustomBlock, self).__init__()
        self.attn = CustomMHA(block_layer.self_attention)
        self.ln_2 = block_layer.ln_2
        self.mlp = block_layer.mlp
        self.dropout = block_layer.dropout

    def forward(self, Xs):
        # x is usual input, zs are ln_1(x)
        x, z1, z2, z3, z = Xs
        x = x + self.dropout(self.attn([z1, z2, z3, z]))
        x = x + self.mlp(self.ln_2(x))
        return x


class NewViT(torch.nn.Module):
    def __init__(self, ViT):
        super(NewViT, self).__init__()

        self.encoder = ViT.encoder.layers
        self.ln = ViT.encoder.ln
        self.heads = ViT.heads

    def forward(self, X):
        for idx, block in enumerate(self.encoder):
            if idx == 0:
                x = block(X)
            else:
                x = block(x)
        x = self.ln(x)
        x = x[:, 0]
        x = self.heads(x)
        return x

def get_jacobian(net, data, BATCH_SIZE=10, IDX=0):

    with torch.no_grad():
        def single_net(x):
            O = net(x).squeeze(0)[IDX:IDX+BATCH_SIZE]
            return O
        return vmap(jacrev(single_net))(data)



def verify_tnfa(LAYER_IDX, head_idx, imagenet_path):

    net = models.vit_b_32(weights="DEFAULT")
    trainloader, testloader, NUM_CLASSES = dataset.get_imagenet(batch_size=1, path=imagenet_path)

    net.eval()
    net.cuda()

    enc_dropout =  net.encoder.dropout

    for idx, layer in enumerate(list(net.encoder.layers)):
        if idx == LAYER_IDX:
            encoder = layer
            attn = encoder.self_attention
            ln_1 = encoder.ln_1
            break

    for idx, param in enumerate(attn.parameters()):
        if idx == 0:
            W = param.data

    n_embed = W.shape[0] // 3

    MQ = W[:n_embed, :]
    MK = W[n_embed*1:n_embed*2, :]
    MV = W[n_embed*2:, :]

    set_head_idx(head_idx, attn.num_heads)

    H = attn.num_heads
    C, _ = MQ.shape

    MQ = MQ.view(H, C // H, -1)
    MQ = MQ[HEAD_IDX, :, :]

    MK = MK.view(H, C // H, -1)
    MK = MK[HEAD_IDX, :, :]

    MV = MV.view(H, C // H, -1)
    MV = MV[HEAD_IDX, :, :]

    MQ = MQ.T @ MQ * 1/len(MQ)
    MK = MK.T @ MK * 1/len(MK)
    MV = MV.T @ MV * 1/len(MV)

    new_block = CustomBlock(encoder)

    new_net = deepcopy(net)

    new_vit = NewViT(new_net)
    new_vit.encoder[LAYER_IDX] = new_block
    new_vit.encoder = new_vit.encoder[LAYER_IDX:]

    new_vit.cuda()
    new_vit.eval()

    NUM_CLASSES = 1000
    GQ, GK, GV = 0, 0, 0
    MAX_IDX = 10

    labelset = set()
    completed_count = 0

    for idx, batch in enumerate(trainloader):
        inputs, targets = batch
        label = targets[0].numpy().item()
        if label in labelset:
            continue
        else:
            labelset.add(label)

        o = net._process_input(inputs.cuda())


        n = o.shape[0]

        batch_class_token = net.class_token.expand(n, -1, -1)
        o = torch.cat([batch_class_token, o], dim=1)

        o = o + net.encoder.pos_embedding
        o = enc_dropout(o)

        for t_idx, layer in enumerate(list(net.encoder.layers)):
            if t_idx < LAYER_IDX:
                o = layer(o)

        with torch.no_grad():
            z1 = ln_1(o).unsqueeze(0)
            z2 = deepcopy(z1)
            z3 = deepcopy(z1)
            z = deepcopy(z1)

        BATCH_SIZE = 100
        IDX = 0
        num_batches = NUM_CLASSES // BATCH_SIZE
        for _ in range(num_batches):
            J = get_jacobian(new_vit, [o.unsqueeze(0), z1, z2, z3, z],
                             BATCH_SIZE=BATCH_SIZE,
                             IDX=IDX)
            JQ = J[1][0]
            JQ = torch.permute(JQ, [1, 2, 0, 3])[0]

            gq = torch.einsum('pcd, pcD -> pdD', JQ, JQ)
            gq = torch.mean(gq, dim=0)
            GQ += gq

            JK = J[2][0]
            JK = torch.permute(JK, [1, 2, 0, 3])[0]

            gk = torch.einsum('pcd, pcD -> pdD', JK, JK)
            gk = torch.mean(gk, dim=0)
            GK += gk

            JV = J[3][0]
            JV = torch.permute(JV, [1, 2, 0, 3])[0]

            gv = torch.einsum('pcd, pcD -> pdD', JV, JV)
            gv = torch.mean(gv, dim=0)
            GV += gv
            IDX += BATCH_SIZE

        completed_count += 1
        print("Completed AGOP for example", completed_count, "out of", MAX_IDX)
        if completed_count == MAX_IDX:
            break

    GQ = sqrt(GQ)
    GK = sqrt(GK)
    GV = sqrt(GV)

    cq = get_item(correlation(MQ, GQ))
    ck = get_item(correlation(MK, GK))
    cv = get_item(correlation(MV, GV))

    print("Layer", LAYER_IDX, "Head", HEAD_IDX, "Query Correlation: ", cq)
    print("Layer", LAYER_IDX, "Head", HEAD_IDX, "Key Correlation: ", ck)
    print("Layer", LAYER_IDX, "Head", HEAD_IDX, "Value Correlation: ", cv)

    return cq, ck, cv


def get_item(x):
    return x.cpu().data.numpy().item()

def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def main():
    fname = None #PATH TO CORRELATION LOG FILE
    outf = open(fname, 'w')

    IMAGENET_PATH = None #PATH TO IMAGENET DATA DIR

    # Adjust l_idx for num layers
    # Adjust h_idx for num heads per layer
    for l_idx in range(12):
        for h_idx in range(12):
            cq, ck, cv = verify_tnfa(LAYER_IDX=l_idx, head_idx=h_idx, imagenet_path=IMAGENET_PATH)
            print("Layer " + str(l_idx) + ',' + "Head " + str(h_idx) + "," + \
                  str(cq) + ',' + str(ck) + ',' + str(cv), file=outf, flush=True)

if __name__ == "__main__":
    main()
