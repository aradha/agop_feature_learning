import os

import torch
from tqdm import tqdm
from functorch import jacrev, vmap

def matrix_sqrt(M):
    device = M.device
    S, V = torch.linalg.eigh(M.cuda())
    S[S<0] = 0
    S = torch.diag(S**0.5)
    return (V @ S @ V.T).to(device)

def corr(A_, B_):
    assert(A_.shape == B_.shape)

    A = A_ - A_.mean()
    B = B_ - B_.mean()

    normA = torch.nan_to_num(torch.linalg.norm(A.reshape(-1)))
    normB = torch.nan_to_num(torch.linalg.norm(B.reshape(-1)))
    out = (torch.sum(A*B)/(normA*normB))
    return torch.nan_to_num(out).item()

def measure_rnn_nfa(net, x, y):

    def get_jacobian(x_, hidden_, i2h_perturb_, i2o_perturb_):
        def fnet_perturbs(i2h_perturb__, i2o_perturb__):
            return net(x_, hidden_, i2h_perturb__, i2o_perturb__)[0]
        return jacrev(fnet_perturbs, argnums=(0,1))(i2h_perturb_, i2o_perturb_)

    hidden = net.init_hidden(len(x)).to(x.device)
    hidden_size = hidden.shape[1]

    i2h_grads = []
    i2o_grads = []
    for i in range(x.size(1)):
        xi = x[:,i]
        i2h_perturb = torch.zeros(xi.size(0), 1, hidden_size*2).to(x.device)
        i2o_perturb = torch.zeros(xi.size(0), 1, hidden_size*2).to(x.device)
        i2h_grad, i2o_grad = vmap(get_jacobian)(xi.unsqueeze(1), hidden.unsqueeze(1), i2h_perturb, i2o_perturb)
        i2h_grads.append(i2h_grad.squeeze())
        i2o_grads.append(i2o_grad.squeeze())

        output, hidden = net(xi, hidden)
    return i2h_grads, i2o_grads

def get_rnn_corrs(net, prefix, randomTrainingExample, sum_before_outer=False, sqrtAGOP=False):
    num_test_nfa = 1
    i2h_agop = 0.
    i2o_agop = 0.
    for _ in tqdm(range(num_test_nfa)):
        # i2x_grads_i : list of (1, num_out, 1, d_in)
        i2h_grads_i, i2o_grads_i = measure_rnn_nfa(net, *randomTrainingExample())
        n, o, d = i2h_grads_i[0].shape

        if sum_before_outer:
            i2h_grads_i = sum(i2h_grads_i).reshape(n*o, d) # (batch_size, num_out, d_in)
            i2o_grads_i = sum(i2o_grads_i).reshape(n*o, d) # (batch_size, num_out, d_in)
        else:
            i2h_grads_i = torch.stack(i2h_grads_i, dim=0).squeeze() # (T, batch_size, num_out, d_in)
            i2h_grads_i = i2h_grads_i.reshape(-1,i2h_grads_i.shape[-1])

            i2o_grads_i = torch.stack(i2o_grads_i, dim=0).squeeze() # (T, batch_size, num_out, d_in)
            i2o_grads_i = i2o_grads_i.reshape(-1,i2o_grads_i.shape[-1])

        i2h_agop += i2h_grads_i.T@i2h_grads_i
        i2o_agop += i2o_grads_i.T@i2o_grads_i

    def get_corrs(i2h_agop, i2o_agop, net, prefix):
        i2h_nfm = net.i2h.weight # (k, d)
        i2h_nfm = i2h_nfm.T@i2h_nfm

        torch.save(i2h_nfm.cpu(), os.path.join("saved_mats", prefix + "_i2h_nfm.pt"))

        i2o_nfm = net.i2o.weight # (k, d)
        i2o_nfm = i2o_nfm.T@i2o_nfm

        torch.save(i2o_nfm.cpu(), os.path.join("saved_mats", prefix + "_i2o_nfm.pt"))

        corr_i2h = corr(i2h_agop, i2h_nfm)
        corr_i2o = corr(i2o_agop, i2o_nfm)

        torch.save(i2h_agop.cpu(), os.path.join("saved_mats", prefix + "_i2h_agop.pt"))
        torch.save(i2o_agop.cpu(), os.path.join("saved_mats", prefix + "_i2o_agop.pt"))

        return corr_i2h, corr_i2o
    
    if sqrtAGOP:
        i2h_agop = matrix_sqrt(i2h_agop)
        i2o_agop = matrix_sqrt(i2o_agop)

    nfa_corrs = get_corrs(i2h_agop, i2o_agop, net, prefix)
    print("nfa_corrs, i2h, i2o:", nfa_corrs)
    return nfa_corrs

def measure_gru_nfa(net, x, y):
    n_perturbs=6
    def get_jacobian(x_, hidden_, p1_, p2_, p3_, p4_, p5_, p6_):
        def fnet_perturbs(p1__, p2__, p3__, p4__, p5__, p6__):
            return net(x_, hidden_, p1__, p2__, p3__, p4__, p5__, p6__)[0]
        return jacrev(fnet_perturbs, argnums=(0,1,2,3,4,5))(p1_, p2_, p3_, p4_, p5_, p6_)

    hidden = net.init_hidden(len(x)).to(x.device)
    hidden_size = hidden.shape[1]

    grads = [[] for _ in range(n_perturbs)]
    for i in range(x.size(1)):
        xi = x[:,i]
        perturbs = [torch.zeros(len(xi), 1, hidden_size).to(x.device) for _ in range(6)]

        jacobians = vmap(get_jacobian)(xi.unsqueeze(1), hidden.unsqueeze(1), *perturbs)
        for i in range(n_perturbs):
            grads[i].append(jacobians[i].squeeze())

        _, hidden = net(xi, hidden)
    return grads

def get_gru_corrs(net, prefix, randomTrainingExample, sum_before_outer=False, sqrtAGOP=False):
    num_test_nfa = 1
    n_perturbs = 6
    agops = [0. for _ in range(n_perturbs)]
    for _ in tqdm(range(num_test_nfa)):
        grads = measure_gru_nfa(net, *randomTrainingExample())
        for i, gs in enumerate(grads):
            if sum_before_outer:
                g_ = sum(gs)
                n, o, d = g_.shape # (batch_size, num_out, d_in)
                g = g_.reshape(n*o, d) 
                agops[i] += g.T @ g
            else:
                g_ = torch.stack(gs)
                g = g_.squeeze() # (T, batch_size, num_out, in_dim)
                _, _, _, dim = g.shape
                g = g.reshape(-1, dim)
                agops[i] += g.T @ g
    if sqrtAGOP:
        for i in range(len(agops)):
            agops[i] = matrix_sqrt(agops[i])

    corrs = []

    Wz_nfm = net.Wz.weight
    Wz_nfm = Wz_nfm.T @ Wz_nfm

    Wr_nfm = net.Wr.weight
    Wr_nfm = Wr_nfm.T @ Wr_nfm

    Wh_nfm = net.Wh.weight
    Wh_nfm = Wh_nfm.T @ Wh_nfm

    Uz_nfm = net.Uz.weight
    Uz_nfm = Uz_nfm.T @ Uz_nfm

    Ur_nfm = net.Ur.weight
    Ur_nfm = Ur_nfm.T @ Ur_nfm

    Uh_nfm = net.Uh.weight
    Uh_nfm = Uh_nfm.T @ Uh_nfm

    torch.save(Wz_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Wz_nfm.pt'))
    torch.save(agops[0].cpu(), os.path.join("saved_mats", prefix + f'_Wz_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Wz_nfm, agops[0]))

    torch.save(Wr_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Wr_nfm.pt'))
    torch.save(agops[1].cpu(), os.path.join("saved_mats", prefix + f'_Wr_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Wr_nfm, agops[1]))

    torch.save(Wh_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Wh_nfm.pt'))
    torch.save(agops[2].cpu(), os.path.join("saved_mats", prefix + f'_Wh_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Wh_nfm, agops[2]))

    torch.save(Uz_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Uz_nfm.pt'))
    torch.save(agops[3].cpu(), os.path.join("saved_mats", prefix + f'_Uz_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Uz_nfm, agops[3]))

    torch.save(Ur_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Ur_nfm.pt'))
    torch.save(agops[4].cpu(), os.path.join("saved_mats", prefix + f'_Ur_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Ur_nfm, agops[4]))

    torch.save(Uh_nfm.cpu(), os.path.join("saved_mats", prefix + f'_Uh_nfm.pt'))
    torch.save(agops[5].cpu(), os.path.join("saved_mats", prefix + f'_Uh_agop_sqrt_{sqrtAGOP}.pt'))
    corrs.append(corr(Uh_nfm, agops[5]))

    print("nfa_corrs:", corrs)
    return corrs
