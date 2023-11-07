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
print("Square root AGOP:",sqrtAGOP)
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


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)

        torch.nn.init.xavier_uniform_(self.i2h.weight, gain=0.05)
        torch.nn.init.xavier_uniform_(self.i2o.weight, gain=0.05)

        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, category, x, hidden, i2h_perturb=None, i2o_perturb=None):
        
        input_combined = torch.cat((category, x, hidden), 1) 
        if i2h_perturb is not None:
            #print("i2h","perturb", i2h_perturb.shape, "combined", input_combined.shape)
            input_combined_h = input_combined + i2h_perturb
        else: 
            input_combined_h = input_combined
        if i2o_perturb is not None:
            #print("i2o",i2o_perturb.shape, input_combined.shape)
            input_combined_o = input_combined + i2o_perturb
        else: 
            input_combined_o = input_combined

        hidden = self.i2h(input_combined_h)
        output = self.i2o(input_combined_o)

        output_combined = torch.cat((hidden, output), 1) 
        #output_combined = nn.GELU()(output_combined)

        output = self.o2o(output_combined)
        output = self.softmax(output)
        return output, hidden

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
        fname = f'saved_models/init_model_lr_{learning_rate}_dict.pt'
    else:
        fname = f'saved_models/trained_model_lr_{learning_rate}_dict.pt'
    torch.save(model.state_dict(), fname)
    return 

def measure_nfa(net, category_tensor, input_line_tensor, target_line_tensor):
    

    def get_jacobian(x_, hidden_, i2h_perturb_, i2o_perturb_):
        def fnet_perturbs(i2h_perturb__, i2o_perturb__):
            return net(category_tensor, x_, hidden_, i2h_perturb__, i2o_perturb__)[0]
        return jacrev(fnet_perturbs, argnums=(0,1))(i2h_perturb_, i2o_perturb_) 

    hidden = net.initHidden().to(category_tensor.device)
    hidden_size = hidden.shape[1]

    i2h_grads = []
    i2o_grads = []
    for i in range(input_line_tensor.size(0)):
        x = input_line_tensor[i]
        input_size = x.shape[1]
        #if i==0:
            #print("n_categories",n_categories,"input_size",input_size, "hidden_size", hidden_size)
        i2h_perturb = torch.zeros(1, n_categories + input_size + hidden_size).to(x.device)
        i2o_perturb = torch.zeros(1, n_categories + input_size + hidden_size).to(x.device)
        i2h_grad, i2o_grad = get_jacobian(x, hidden, i2h_perturb, i2o_perturb)
        i2h_grads.append(i2h_grad)
        i2o_grads.append(i2o_grad)

        output, hidden = net(category_tensor, x, hidden)
    return i2h_grads, i2o_grads

def get_nfa_corrs(net, prefix, sum_before_outer=False):
    num_test_nfa = 100
    i2h_agop = 0. 
    i2o_agop = 0.
    for _ in tqdm(range(num_test_nfa)):
        # i2x_grads_i : list of (1, num_out, 1, d_in)
        i2h_grads_i, i2o_grads_i = measure_nfa(net, *randomTrainingExample())
        
        if sum_before_outer: 
            i2h_grads_i = sum(i2h_grads_i).squeeze() # (num_out, d_in)
            i2o_grads_i = sum(i2o_grads_i).squeeze() # (num_out, d_in)
        else:
            i2h_grads_i = torch.stack(i2h_grads_i, dim=0).squeeze() # (T, num_out, d_in)
            i2h_grads_i = i2h_grads_i.reshape(-1,i2h_grads_i.shape[-1])

            i2o_grads_i = torch.stack(i2o_grads_i, dim=0).squeeze() # (T, num_out, d_in)
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

learning_rate = 5e-4
width = 512
rnn = RNN(n_letters, width, n_letters)
save_model(rnn, init=True, learning_rate=learning_rate)
rnn.cuda()
print("LR:", learning_rate, "width:",width)

# get init corrs
sum_before_outer = False
print("Sum before outer:",sum_before_outer)
corrh_init, corro_init = get_nfa_corrs(rnn, "init", sum_before_outer=sum_before_outer)

n_iters = 250000
print_every = 1000
plot_every = 1000
all_losses = []
total_loss = 0 # Reset every ``plot_every`` ``iters``

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
corrh_end, corro_end = get_nfa_corrs(best_model, "end", sum_before_outer=sum_before_outer)


with open(f'results/nfa_corrs_lr_{learning_rate}_sqrtAGOP_{sqrtAGOP}.txt', "w") as f:
    f.write(f'init i2h: {corrh_init}, init i2o: {corro_init}, end i2h: {corrh_end}, end i2o: {corro_end}') 

with open(f'results/train_losses_lr_{learning_rate}_sqrtAGOP_{sqrtAGOP}.txt', "w") as f:
    f.writelines([str(x) + "," for x in all_losses])

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

save_model(rnn.cpu(), init=False, learning_rate=learning_rate)
