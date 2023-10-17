import numpy as np
import torch
import random
import dataset
import neural_model
from torch.linalg import norm
from functorch import jacrev, vmap

SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def get_name(dataset_name, configs):
    name_str = dataset_name
    for key in configs:
        name_str += key + ':' + str(configs[key] + ':')
    name_str += 'nn'
    return name_str


def load_nn(path, width, depth, dim, num_classes, layer_idx=0,
            remove_init=False, act_name='relu'):

    if remove_init:
        suffix = path.split('/')[-1]
        prefix = './saved_nns/'

        init_net = neural_model.Net(dim, width=width, depth=depth,
                                    num_classes=num_classes,
                                    act_name=act_name)
        d = torch.load(prefix + 'init_' + suffix)
        init_net.load_state_dict(d['state_dict'])
        init_params = [p for idx, p in enumerate(init_net.parameters())]

    net = neural_model.Net(dim, width=width, depth=depth,
                           num_classes=num_classes,
                           act_name=act_name)

    d = torch.load(path)
    net.load_state_dict(d['state_dict'])

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            if remove_init:
                M0 = init_params[idx].data.numpy()
                M -= M0
            break

    M = M.T @ M * 1/len(M)

    return net, M


def load_init_nn(path, width, depth, dim, num_classes, layer_idx=0, act_name='relu'):
    suffix = path.split('/')[-1]
    prefix = './saved_nns/'

    net = neural_model.Net(dim, width=width, depth=depth,
                                num_classes=num_classes, act_name=act_name)
    d = torch.load(prefix + 'init_' + suffix)
    net.load_state_dict(d['state_dict'])

    for idx, p in enumerate(net.parameters()):
        if idx == layer_idx:
            M = p.data.numpy()
            print(M.shape)
            break

    M = M.T @ M * 1/len(M)
    return net, M



def get_layer_output(net, trainloader, layer_idx=0):
    net.eval()
    out = []
    for idx, batch in enumerate(trainloader):
        data, labels = batch
        if layer_idx == 0:
            out.append(data.cpu())
        elif layer_idx == 1:
            o = neural_model.Nonlinearity()(net.first(data))
            out.append(o.cpu())
        elif layer_idx > 1:
            o = net.first(data)
            for l_idx, m in enumerate(net.middle):
                o = m(o)
                if l_idx + 1 == layer_idx:
                    o = neural_model.Nonlinearity()(o)
                    out.append(o.cpu())
                    break
    out = torch.cat(out, dim=0)
    net.cpu()
    return out


def build_subnetwork(net, dim, width, depth, num_classes,
                     layer_idx=0, random_net=False, act_name='relu'):

    net_ = neural_model.Net(dim, depth=depth - layer_idx,
                            width=width, num_classes=num_classes,
                            act_name=act_name)

    params = [p for idx, p in enumerate(net.parameters())]
    if not random_net:
        for idx, p_ in enumerate(net_.parameters()):
            p_.data = params[idx + layer_idx].data

    return net_


def get_jacobian(net, data):
    with torch.no_grad():
        return vmap(jacrev(net))(data).transpose(0, 2).transpose(0, 1)


def egop(net, dataset, centering=False):
    device = torch.device('cuda')
    bs = 1000
    batches = torch.split(dataset, bs)
    net = net.cuda()
    G = 0

    Js = []
    for batch_idx, data in enumerate(batches):
        data = data.to(device)
        print("Computing Jacobian for batch: ", batch_idx, len(batches))
        J = get_jacobian(net, data)
        Js.append(J.cpu())

        # Optional for stopping EGOP computation early
        #if batch_idx > 30:
        #    break
    Js = torch.cat(Js, dim=-1)
    if centering:
        J_mean = torch.mean(Js, dim=-1).unsqueeze(-1)
        Js = Js - J_mean

    Js = torch.transpose(Js, 2, 0)
    Js = torch.transpose(Js, 1, 2)
    print(Js.shape)
    batches = torch.split(Js, bs)
    for batch_idx, J in enumerate(batches):
        print(batch_idx, len(batches))
        m, c, d = J.shape
        J = J.cuda()
        G += torch.einsum('mcd,mcD->dD', J, J).cpu()
        del J
    G = G * 1/len(Js)

    return G


def correlate(M, G):
    M = M.double()
    G = G.double()
    normM = norm(M.flatten())
    normG = norm(G.flatten())

    corr = torch.dot(M.flatten(), G.flatten()) / (normM * normG)
    return corr


def read_configs(path):
    tokens = path.strip().split(':')
    print(tokens)
    act_name = 'relu'
    for idx, t in enumerate(tokens):
        if t == 'width':
            width = eval(tokens[idx+1])
        if t == 'depth':
            depth = eval(tokens[idx+1])
        if t == 'act':
            act_name = tokens[idx+1]

    return width, depth, act_name


def verify_NFA(path, dataset_name, feature_idx=None, layer_idx=0):
    remove_init = True
    random_net = False

    if dataset_name == 'celeba':
        NUM_CLASSES = 2
        FEATURE_IDX = feature_idx
        SIZE = 96
        c = 3
        dim = c * SIZE * SIZE
    elif dataset_name == 'svhn' or dataset_name == 'cifar':
        NUM_CLASSES = 10
        SIZE = 32
        c = 3
        dim = c * SIZE * SIZE
    elif dataset_name == 'cifar_mnist':
        NUM_CLASSES = 10
        c = 3
        SIZE = 32
        dim = c * SIZE * SIZE * 2
    elif dataset_name == 'stl_star':
        NUM_CLASSES = 2
        c = 3
        SIZE = 96
        dim = c * SIZE * SIZE

    width, depth, act_name = read_configs(path)

    net, M = load_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                     remove_init=remove_init, act_name=act_name)
    net0, M0 = load_init_nn(path, width, depth, dim, NUM_CLASSES, layer_idx=layer_idx,
                            act_name=act_name)
    subnet = build_subnetwork(net, M.shape[0], width, depth, NUM_CLASSES, layer_idx=layer_idx,
                              random_net=random_net, act_name=act_name)

    init_correlation = correlate(torch.from_numpy(M),
                                 torch.from_numpy(M0))

    print("Init Net Feature Matrix Correlation: " , init_correlation)

    if dataset_name == 'celeba':
        trainloader, valloader, testloader = dataset.get_celeba(FEATURE_IDX,
                                                                num_train=20000,
                                                                num_test=1)
    elif dataset_name == 'svhn':
        trainloader, valloader, testloader = dataset.get_svhn(num_train=1000,
                                                              num_test=1)
    elif dataset_name == 'cifar':
        trainloader, valloader, testloader = dataset.get_cifar(num_train=1000,
                                                               num_test=1)

    elif dataset_name == 'cifar_mnist':
        trainloader, valloader, testloader = dataset.get_cifar_mnist(num_train_per_class=1000,
                                                                     num_test_per_class=1)
    elif dataset_name == 'stl_star':
        trainloader, valloader, testloader = dataset.get_stl_star(num_train=1000,
                                                                  num_test=1)
    out = get_layer_output(net, trainloader, layer_idx=layer_idx)
    G = egop(subnet, out, centering=True)
    G2 = egop(subnet, out, centering=False)

    centered_correlation = correlate(torch.from_numpy(M), G)
    uncentered_correlation = correlate(torch.from_numpy(M), G2)
    print("Full Matrix Correlation Centered: " , centered_correlation)
    print("Full Matrix Correlation Uncentered: " , uncentered_correlation)

    return init_correlation, centered_correlation, uncentered_correlation

def main():

    path = ''  # Path to saved neural net model
    idxs = [0, 1, 2] # Layers for which to compute EGOP
    init, centered, uncentered = [], [], []
    for idx in idxs:
        results = verify_NFA(path, 'svhn', layer_idx=idx)
        i, c, u = results
        init.append(i.numpy().item())
        centered.append(c.numpy().item())
        uncentered.append(u.numpy().item())
    for idx in idxs:
        print("Layer " + str(idx), init[idx], centered[idx], uncentered[idx])


if __name__ == "__main__":
    main()
