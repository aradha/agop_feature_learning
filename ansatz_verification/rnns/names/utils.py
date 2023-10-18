import time
import math

import torch

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def corr(A_, B_):
    assert(A_.shape == B_.shape)

    A = A_ - A_.mean()
    B = B_ - B_.mean()

    normA = torch.nan_to_num(torch.linalg.norm(A.reshape(-1)))
    normB = torch.nan_to_num(torch.linalg.norm(B.reshape(-1)))
    out = (torch.sum(A*B)/(normA*normB))
    return torch.nan_to_num(out).item()
