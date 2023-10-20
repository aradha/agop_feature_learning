import os
import sys
import time

import neural_tangents as nt

import torch
import numpy as np
import jax 
from jax import random
import jax.numpy as jnp
from jax import grad, jit, vmap, jacrev
from einops import rearrange

from torch.nn.functional import pad as torch_pad
import torchvision
import torchvision.transforms as transforms
from math import sqrt, pi
from tqdm import tqdm
import time

import utils

def get_grads(kernel_fn, alphas, X_train_M, Xs, M, num_classes, ps):
    def get_solo_grads(sol, X, x):
        def egop_fn(z):
            if M is not None:
                z = utils.multiply_patches(z, M, ps)
            K = kernel_fn(X, z, fn='ntk').ntk
            return (sol @ K).squeeze()
        grads = jax.vmap(jax.grad(egop_fn))(jnp.expand_dims(x,1)).squeeze()
        grads = jnp.nan_to_num(grads)
        return grads 

    n, P, Q, c = X_train_M.shape
    p = P//ps
    q = Q//ps
    w, h = (ps,ps)
    s = len(Xs)

    chunk = 250
    leftover_bool = int(n%chunk>0)
    train_batches = jnp.array_split(jnp.arange(n), n//chunk + leftover_bool)

    egop = 0
    sol = jnp.array(alphas.T)
    for o in tqdm(range(num_classes)):
        grads = 0
        for btrain in train_batches:
            grads += get_solo_grads(sol[o,btrain], X_train_M[btrain], Xs)
        grads = grads.reshape(-1, p, w, q, h, c) # n, p, w, q, h, c
        G = torch.from_numpy(np.array(grads))
        G = G.transpose(2, 3).transpose(-2, -1).transpose(-3, -2)
        n, _, _, c, w, h = G.shape
        G = G.reshape(-1, c*w*h)
        egop += G.T @ G/s

    return egop

