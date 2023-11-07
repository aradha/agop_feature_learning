import os
import torch
import numpy as np
from model import GPTConfig, GPT
import torch.nn as nn
from copy import deepcopy
from functorch import jacrev, vmap
from torch.linalg import norm, svd
from torch.nn import functional as F
import math
import random

SEED = 1717
from tqdm import tqdm
import torch.backends.cudnn as cudnn

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
cudnn.benchmark = True


HEAD_IDX = None
HEAD_IDXS = None
PERMUTATION = None

def set_head_idx(head_idx, num_heads=4):
    global HEAD_IDX, HEAD_IDXS, PERMUTATION
    HEAD_IDX = head_idx
    HEAD_IDXS = list((i for i in range(num_heads) if i != HEAD_IDX))
    PERMUTATION = list([i+1 for i in range(num_heads-1)])
    PERMUTATION.insert(HEAD_IDX, 0)
    PERMUTATION = torch.LongTensor(PERMUTATION).cuda()

def load_model(ckpt_path, device='cuda'):
    print("MODEL LOADED: ", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model

def get_batch(data, batch_size=64, block_size=256):
    device = 'cuda'
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def load_data(data_dir):
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    return train_data


class CustomAttention(nn.Module):

    def __init__(self, attn):
        super().__init__()
        # key, query, value projections for all heads, but in a batch
        self.c_attn = attn.c_attn
        # output projection
        self.c_proj = attn.c_proj
        # regularization
        self.attn_dropout = attn.attn_dropout
        self.resid_dropout = attn.resid_dropout
        self.n_head = attn.n_head
        self.n_embd = attn.n_embd
        self.dropout = attn.dropout
        self.bias = attn.bias

    def forward(self, X):

        x1, x2, x3, x = X

        B, T, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, _, _  = self.c_attn(x1).split(self.n_embd, dim=2)
        _, k, _  = self.c_attn(x2).split(self.n_embd, dim=2)
        _, _, v  = self.c_attn(x3).split(self.n_embd, dim=2)
        qt, kt, vt = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        kt = kt.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        qt = qt.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        vt = vt.view(B, T, self.n_head, self.n_embd // self.n_head).transpose(1, 2) # (B, nh, T, hs)


        kf = torch.cat([k[:, HEAD_IDX:HEAD_IDX+1, :, :], kt[:, HEAD_IDXS, :, :]], dim=1)
        qf = torch.cat([q[:, HEAD_IDX:HEAD_IDX+1, :, :], qt[:, HEAD_IDXS, :, :]], dim=1)
        vf = torch.cat([v[:, HEAD_IDX:HEAD_IDX+1, :, :], vt[:, HEAD_IDXS, :, :]], dim=1)

        # Reorganize heads
        kf = kf.index_select(1, PERMUTATION)
        qf = qf.index_select(1, PERMUTATION)
        vf = vf.index_select(1, PERMUTATION)

        att = (qf @ kf.transpose(-2, -1)) * (1.0 / np.sqrt(kt.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ vf # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y


class CustomBlock(torch.nn.Module):

    def __init__(self, block_layer):
        super(CustomBlock, self).__init__()

        self.attn = CustomAttention(block_layer.attn)
        self.ln_2 = block_layer.ln_2
        self.mlp = block_layer.mlp

    def forward(self, Xs):
        # x is usual input, z is ln_1(x)
        x, z1, z2, z3, z = Xs
        x = x + self.attn([z1, z2, z3, z])
        x = x + self.mlp(self.ln_2(x))
        return x


class NewGPT(torch.nn.Module):
    def __init__(self, GPT):
        super(NewGPT, self).__init__()

        self.h = GPT.transformer.h
        self.ln_f = GPT.transformer.ln_f
        self.lm_head = GPT.lm_head

    def forward(self, X):
        #x, z = X
        for idx, block in enumerate(self.h):
            if idx == 0:
                x = block(X)
            else:
                x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


def get_jacobian(net, data, BATCH_SIZE=16, IDX=0, OUT_SIZE=100, OUT_IDX=0):
    with torch.no_grad():
        def single_net(x):
            O = net(x).squeeze(0)[IDX:IDX+BATCH_SIZE, OUT_IDX:OUT_IDX+OUT_SIZE]
            return O
        return vmap(jacrev(single_net))(data)


def min_max(M):
    return (M - M.min()) / (M.max() - M.min())


def correlation(M1, M2):

    M1 /= M1.max()
    M2 /= M2.max()
    M1 -= M1.mean()
    M2 -= M2.mean()

    norm1 = norm(M1.flatten())
    norm2 = norm(M2.flatten())
    return torch.sum(M1 * M2) / (norm1 * norm2 + np.finfo('float').eps)


def check_ansatz(layer_idx, head_idx, ckpt_path, data_dir,
                 VOCAB_SIZE=65):

    model = load_model(ckpt_path)
    new_model = deepcopy(model)

    model.eval()
    model.cuda()

    new_model.eval()
    new_model.cuda()

    T = new_model.transformer

    attn = T.h[layer_idx].attn.c_attn
    attn_layer = T.h[layer_idx].attn
    set_head_idx(head_idx, attn_layer.n_head)

    for idx, param in enumerate(attn.parameters()):
        W = param.data

    n_embed = W.shape[0] // 3

    MQ = W[:n_embed, :]
    MK = W[n_embed*1:n_embed*(1+1), :]
    MV = W[n_embed*2:, :]

    H = attn_layer.n_head
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

    T.h[layer_idx] = CustomBlock(T.h[layer_idx])
    T.h = T.h[layer_idx:]

    newGPT = NewGPT(new_model)

    newGPT.eval()
    newGPT.cuda()

    GQ = 0
    GK = 0
    GV = 0
    MAX_IDX = 10

    train_data = load_data(data_dir)
    for idx in range(MAX_IDX):
        print("Running AGOP for Training Example: ", idx)
        x, y = get_batch(train_data)
        x = x[0].unsqueeze(0)

        Q = model.transformer

        with torch.no_grad():
            n, t = x.shape
            pos = torch.arange(0, t, dtype=torch.long, device='cuda')
            z = Q.drop(Q.wpe(pos) + Q.wte(x))

            for i in range(layer_idx):
                z = Q.h[i](z)

        with torch.no_grad():
            ln_z1 = Q.h[layer_idx].ln_1(z).unsqueeze(0)
            ln_z2 = deepcopy(ln_z1)
            ln_z3 = deepcopy(ln_z1)
            ln_z = deepcopy(ln_z1)
        z = z.cuda()

        B, T, C = ln_z1[0].shape
        BATCH_SIZE = 8
        IDX = 0
        num_batches = T // BATCH_SIZE

        OUT_SIZE = 25
        num_out_batches = VOCAB_SIZE // OUT_SIZE

        for _ in tqdm(range(num_batches)):
            OUT_IDX = 0
            for _ in tqdm(range(num_out_batches)):
                J = get_jacobian(newGPT, [z.unsqueeze(0), ln_z1, ln_z2, ln_z3, ln_z],
                                 BATCH_SIZE=BATCH_SIZE,
                                 IDX=IDX,
                                 OUT_SIZE=OUT_SIZE,
                                 OUT_IDX=OUT_IDX)

                JQ = J[1][0]
                JQ = JQ.reshape(-1, B, T, C)
                JQ = torch.permute(JQ, [1, 2, 0, 3])[0]

                g = torch.einsum('pcd, pcD -> pdD', JQ, JQ)

                g = torch.sum(g, dim=0)
                GQ += g

                JK = J[2][0]
                JK = JK.reshape(-1, B, T, C)
                JK = torch.permute(JK, [1, 2, 0, 3])[0]

                g = torch.einsum('pcd, pcD -> pdD', JK, JK)
                g = torch.sum(g, dim=0)
                GK += g

                JV = J[3][0]
                JV = JV.reshape(-1, B, T, C)
                JV = torch.permute(JV, [1, 2, 0, 3])[0]

                g = torch.einsum('pcd, pcD -> pdD', JV, JV)
                g = torch.sum(g, dim=0)
                GV += g

                IDX += BATCH_SIZE
                OUT_IDX += OUT_SIZE

    GQ = sqrt(GQ)
    GK = sqrt(GK)
    GV = sqrt(GV)

    print(layer_idx, head_idx, "Query Correlation: ", correlation(MQ, GQ))
    print(layer_idx, head_idx, "Key Correlation: ", correlation(MK, GK))
    print(layer_idx, head_idx, "Value Correlation: ", correlation(MV, GV))

    return get_item(correlation(MQ, GQ)), get_item(correlation(MK, GK)), get_item(correlation(MV, GV))

def get_item(x):
    return x.cpu().numpy().item()


def sqrt(G):
    U, s, Vt = svd(G)
    s = torch.pow(s, 1./2)
    G = U @ torch.diag(s) @ Vt
    return G


def main():
    fname = None #PATH FOR LOGGING CORRELATIONS
    outf = open(fname, 'w')

    layers = list(range(6))
    heads = list(range(6))

    VOCAB_SIZE = 65 # VOCAB SIZE FOR LLM
    ckpt_path = None # PATH TO MODEL CHECKPOINT
    data_dir = None # PATH TO DATA DIRECTORY

    for layer in layers:
        for head in heads:
            cq, ck, cv = check_ansatz(layer, head, ckpt_path, data_dir, VOCAB_SIZE=VOCAB_SIZE)
            print("Layer " + str(layer) + ',' + "Head " + str(head) + "," + \
                  str(cq) + ',' + str(ck) + ',' + str(cv), file=outf, flush=True)


if __name__ == "__main__":
    main()
