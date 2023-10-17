import torch
from torch.autograd import Variable
import torch.optim as optim
import time
import neural_model
import numpy as np
from sklearn.metrics import r2_score


def select_optimizer(name, lr, net, weight_decay):
    if name == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)


def train_network(train_loader, val_loader, test_loader, num_classes,
                  name='default_nn', configs=None, regression=False):

    for idx, batch in enumerate(train_loader):
        inputs, labels = batch
        _, dim = inputs.shape
        break

    if configs is not None:
        num_epochs = configs['num_epochs'] + 1
        net = neural_model.Net(dim, width=configs['width'],
                               depth=configs['depth'],
                               num_classes=num_classes,
                               act_name=configs['act'])

        if configs['init'] != 'default':
            for idx, param in enumerate(net.parameters()):
                if idx == 0:
                    init = torch.Tensor(param.size()).normal_().float() * configs['init']
                    param.data = init

        if configs['freeze']:
            for idx, param in enumerate(net.parameters()):
                if idx > 0:
                    param.requires_grad = False


        optimizer = select_optimizer(configs['optimizer'],
                                     configs['learning_rate'],
                                     net,
                                     configs['weight_decay'])
    else:
        num_epochs = 501
        net = neural_model.Net(dim, num_classes=num_classes)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    d = {}
    d['state_dict'] = net.state_dict()
    if name is not None:
        torch.save(d, 'saved_nns/init_' + name + '.pth')


    net.cuda()
    best_val_acc = 0
    best_test_acc = 0
    best_val_loss = np.float("inf")
    best_test_loss = 0

    for i in range(num_epochs):

        train_loss = train_step(net, optimizer, train_loader)
        val_loss = val_step(net, val_loader)
        test_loss = val_step(net, test_loader)
        if regression:
            train_acc = get_r2(net, train_loader)
            val_acc = get_r2(net, val_loader)
            test_acc = get_r2(net, test_loader)
        else:
            train_acc = get_acc(net, train_loader)
            val_acc = get_acc(net, val_loader)
            test_acc = get_acc(net, test_loader)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            net.cpu()
            d = {}
            d['state_dict'] = net.state_dict()
            if name is not None:
                torch.save(d, 'saved_nns/' + name + '.pth')
            net.cuda()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss

        print("Epoch: ", i,
              "Train Loss: ", train_loss, "Test Loss: ", test_loss,
              "Train Acc: ", train_acc, "Test Acc: ", test_acc,
              "Best Val Acc: ", best_val_acc, "Best Val Loss: ", best_val_loss,
              "Best Test Acc: ", best_test_acc, "Best Test Loss: ", best_test_loss)

    net.cpu()

    d = {}
    d['state_dict'] = net.state_dict()
    torch.save(d, 'saved_nns/' + name + '_final.pth')
    return train_acc, best_val_acc, best_test_acc

def train_step(net, optimizer, train_loader):
    net.train()
    start = time.time()
    train_loss = 0.
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs).cuda())
        target = Variable(targets).cuda()
        loss = torch.mean(torch.pow(output - target, 2))
        loss.backward()
        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    end = time.time()
    print("Time: ", end - start)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()
        loss = torch.mean(torch.pow(output - target, 2))
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


def get_acc(net, loader):
    net.eval()
    count = 0
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda())
            target = Variable(targets).cuda()

        preds = torch.argmax(output, dim=-1)
        labels = torch.argmax(target, dim=-1)

        count += torch.sum(labels == preds).cpu().data.numpy()
    return count / len(loader.dataset) * 100


def get_r2(net, loader):
    net.eval()
    count = 0
    preds = []
    labels = []
    for batch_idx, batch in enumerate(loader):
        inputs, targets = batch
        with torch.no_grad():
            output = net(Variable(inputs).cuda()).flatten().cpu().numpy()
            target = Variable(targets).cuda().flatten().cpu().numpy()
            preds.append(output)
            labels.append(target)
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return r2_score(labels, preds)
