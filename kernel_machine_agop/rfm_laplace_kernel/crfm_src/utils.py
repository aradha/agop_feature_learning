import jax
import jax.numpy as jnp
import numpy as np
import neural_tangents as nt
import torch
from einops import rearrange

def multiply_patches(X, M, ps):
    """
    X : (n, p*ps, q*ps, c)
    M : (c*w*h, c*w*h)
    out : (n, p*ps, q*ps, c)
    """
    n = X.shape[0]
    chunk = 5000
    leftover_bool = int(n%chunk>0)
    batches = jnp.array_split(jnp.arange(n), n//chunk + leftover_bool)

    Xs = []
    for i, b in enumerate(batches):
        Xb = X[b]
        m, P, Q, c = Xb.shape
        p = P//ps
        q = Q//ps
        Xb = rearrange(Xb, 'm (p w) (q h) c -> (m p q) (c w h)', p=p, q=q, w=ps, h=ps)
        Xb = Xb @ M
        Xb = rearrange(Xb, '(m p q) (c w h) -> m (p w) (q h) c', m=m, p=p, q=q, c=c, w=ps, h=ps)
        Xs.append(jnp.array(Xb))
    return jnp.concatenate(Xs, axis=0)

def matrix_sqrt(M):
    S, V = torch.linalg.eigh(M.cuda())
    S[S<0] = 0
    S = torch.diag(S**0.5)
    return (V @ S @ V.T).cpu()

def get_batch_size(n1, n2, num_devices):
    n1_ = n1//num_devices
    max_batch_size = 100
    best_batch_size = 1
    for i in range(1,max_batch_size+1):
        if (n1_%i == 0) and (n2%i==0):
            best_batch_size = i
    return best_batch_size


def expand_image(X, ps, padding=False):
    """
    X : (n, c, p, q)
    out : (n, c, p*ps, q*ps)
    """

    if padding:
        pad_sz = ps//2
        pad = (pad_sz,pad_sz,pad_sz,pad_sz)
        X_patched = F.pad(X, pad)
    else:
        X_patched = X

    X_patched = X_patched.unfold(2,ps,1).unfold(3,ps,1) # (n, c, p, q, ps, ps)

    n, c, p, q, _, _ = X_patched.shape
    X_patched = X_patched.transpose(-2,-3) # (n, c, p, ps, q, ps)
    X_expanded = X_patched.reshape(n,c,p*ps,q*ps)
    return X_expanded

def batch_kernel(kernel, X, Z):
    nx = len(X)
    nz = len(Z)
    chunk = 5000
    leftover_bool_x = int(nx%chunk>0)
    leftover_bool_z = int(nz%chunk>0)
    xbatches = jnp.array_split(jnp.arange(nx), nx//chunk + leftover_bool_x)
    zbatches = jnp.array_split(jnp.arange(nz), nz//chunk + leftover_bool_z)

    num_devices = jax.device_count()
    K = [None]*len(xbatches)
    for i, bx in enumerate(xbatches):
        Xb = X[bx]

        Kx = [None]*len(zbatches)
        for j, bz in enumerate(zbatches):

            BS = get_batch_size(len(bx), len(bz), num_devices)
            ntk_fn = nt.batch(kernel,
                               device_count=-1,
                               batch_size=BS,
                               store_on_device=False)

            Zb = Z[bz]

            Kx[j] = ntk_fn(Xb, Zb).ntk

        K[i] = np.concatenate(Kx,axis=1)

    return np.array(np.concatenate(K,axis=0))
