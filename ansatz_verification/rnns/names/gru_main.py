from io import open
import glob
import os
import unicodedata
import string

import torch
import torch.nn as nn
from functorch import jacrev

from utils import *
from trainer import *

from tqdm import tqdm
from copy import deepcopy

torch.manual_seed(0)
random.seed(0)

sqrtAGOP = True
print("Square root AGOP")

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')

print('# categories:', n_categories, all_categories)
print(unicodeToAscii("O'Néàl"))


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        self.Wz = nn.Linear(n_categories + input_size, hidden_size, bias=False)
        self.Wr = nn.Linear(n_categories + input_size, hidden_size, bias=False)
        self.Wh = nn.Linear(n_categories + input_size, hidden_size, bias=False)

        self.Uz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Uh = nn.Linear(hidden_size, hidden_size, bias=False)

        self.Wy = nn.Linear(hidden_size, output_size, bias=False)

        torch.nn.init.xavier_uniform_(self.Wz.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Wr.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Wh.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Uz.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Ur.weight, gain=0.01)
        torch.nn.init.xavier_uniform_(self.Uh.weight, gain=0.01)

    def forward(self, category, x, h, p1=None, p2=None, p3=None, p4=None, p5=None, p6=None):
        
        xc = torch.cat((category, x), 1) 
        if p1 is not None:
            xc_Wz = xc + p1
            xc_Wr = xc + p2
            xc_Wh = xc + p3

            h_Uz = h + p4
            h_Ur = h + p5
            h_Uh = h + p6
        else:
            xc_Wz = xc
            xc_Wr = xc
            xc_Wh = xc

            h_Uz = h
            h_Ur = h
            h_Uh = h

        z = nn.Sigmoid()(self.Wz(xc_Wz) + self.Uz(h_Uz))
        r = nn.Sigmoid()(self.Wr(xc_Wr) + self.Ur(h_Ur))
        ht = nn.Tanh()(self.Wh(xc_Wh) + r * self.Uh(h_Uh))

        h = z*h + (1-z)*ht
        out = nn.LogSoftmax(dim=1)(self.Wy(h))
        return out, h 

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# ``LongTensor`` of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category).cuda()
    input_line_tensor = inputTensor(line).cuda()
    target_line_tensor = targetTensor(line).cuda()
    return category_tensor, input_line_tensor, target_line_tensor

def save_model(model, init, learning_rate):
    if init:
        fname = f'saved_models/gru_init_model_lr_{learning_rate}_dict.pt'
    else:
        fname = f'saved_models/gru_trained_model_lr_{learning_rate}_dict.pt'
    torch.save(model.state_dict(), fname)
    return 

def measure_nfa(net, category_tensor, input_line_tensor, target_line_tensor, n_perturbs):
    
    def get_jacobian(x_, hidden_, p1_, p2_, p3_, p4_, p5_, p6_):
        def fnet_perturbs(p1__, p2__, p3__, p4__, p5__, p6__):
            return net(category_tensor, x_, hidden_, p1__, p2__, p3__, p4__, p5__, p6__)[0]
        return jacrev(fnet_perturbs, argnums=(0,1,2,3,4,5))(p1_, p2_, p3_, p4_, p5_, p6_)

    hidden = net.initHidden().to(category_tensor.device)
    hidden_size = hidden.shape[1]
    
    grads = [[] for _ in range(n_perturbs)]
    for i in range(input_line_tensor.size(0)):
        x = input_line_tensor[i]
        input_size = x.shape[1]

        perturbs = [torch.zeros(input_size + n_categories).to(x.device) for _ in range(3)]
        perturbs += [torch.zeros(hidden_size).to(x.device) for _ in range(3)]

        jacobians = get_jacobian(x, hidden, *perturbs)

        for i in range(n_perturbs):
            grads[i].append(jacobians[i])

        output, hidden = net(category_tensor, x, hidden)
    return grads

def get_nfa_corrs(net, prefix, sum_before_outer=False):
    print("Sum before outer:",sum_before_outer)
    num_test_nfa = 100
    n_perturbs = 6
    agops = [0. for _ in range(n_perturbs)]
    for _ in tqdm(range(num_test_nfa)):
        grads = measure_nfa(net, *randomTrainingExample(), n_perturbs)
        for i, gs in enumerate(grads):
            if sum_before_outer:
                g_ = sum(gs)
                g = g_.squeeze()
                agops[i] += g.T @ g
            else:
                g_ = torch.stack(gs)
                g = g_.squeeze() # (T, num_out, in_dim)
                _, _, dim = g.shape
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

learning_rate = 5e-4
width = 512
rnn = GRU(n_letters, width, n_letters)
save_model(rnn, init=True, learning_rate=learning_rate)
rnn.cuda()

# get init corrs
init_corrs = get_nfa_corrs(rnn, "gru_init")

n_iters = 150000
print_every = 1000
plot_every = 1000
all_losses = []
total_loss = 0 # Reset every ``plot_every`` ``iters``

print("LR:", learning_rate, "width:",width, "epochs:", n_iters)

start = time.time()

best_loss = float("inf")
best_model = None
for iter in range(1, n_iters + 1):
    output, loss = train(rnn, *randomTrainingExample(), learning_rate, n_categories)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

    #if loss < best_loss:
    best_model = deepcopy(rnn)

# get final corrs
end_corrs = get_nfa_corrs(best_model, "gru_end")
max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        hidden = best_model.initHidden().cuda()
        category_tensor = categoryTensor(category).to(hidden.device)
        x = inputTensor(start_letter).to(hidden.device)

        output_name = start_letter

        for i in range(max_length):
            output, hidden = best_model(category_tensor, x[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            x = inputTensor(letter).to(hidden.device)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

samples('Russian', 'RUS')
samples('German', 'GER')
samples('Spanish', 'SPA')
samples('Scottish', 'SCO')
samples('English', 'ENG')
samples('French', 'FRE')
samples('Italian', 'ITA')

with open(f'results/gru_nfa_corrs_lr_{learning_rate}_sqrtAGOP_{sqrtAGOP}.txt', "w") as f:
    f.writelines([str(x) + "," for x in init_corrs])
    f.write("\n")
    f.writelines([str(x) + "," for x in end_corrs])
with open(f'results/gru_train_losses_lr_{learning_rate}_sqrtAGOP_{sqrtAGOP}.txt', "w") as f:
    f.writelines([str(x) + "," for x in all_losses])

save_model(rnn.cpu(), init=False, learning_rate=learning_rate)
