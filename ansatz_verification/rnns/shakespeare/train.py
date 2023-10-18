#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
import argparse

from tqdm import tqdm
import functools

import utils
from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="basic")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--print_every', type=int, default=50)
argparser.add_argument('--hidden_size', type=int, default=64)
argparser.add_argument('--learning_rate', type=float, default=2e-4)
argparser.add_argument('--cuda', type=bool, default=1)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=64)
argparser.add_argument('--shuffle', action='store_true')
args = argparser.parse_args()

for n_, v_ in args.__dict__.items():
        print(f"{n_:<20} : {v_}")

if args.cuda:
    print("Using CUDA")

file, file_len = read_file(args.filename)

def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len - 1)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if args.cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    if args.cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0.

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        #print("output",output.shape, "target[:,c]", target[:,c].shape)
        loss += criterion(output, target[:,c])
        #print("loss", loss)

    loss.backward()
    decoder_optimizer.step()
    #print("loss.data", loss.data, loss.data.shape)

    return loss.data / args.chunk_len

def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + f'_{args.model}_{args.learning_rate}.pt'
    save_filename = os.path.join('saved_models', save_filename)
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)

# Initialize models and start training
if args.model=="basic":
    decoder = RNN(
        n_characters,
        args.hidden_size,
        n_characters,
    )
    corr_fn = utils.get_rnn_corrs 
elif args.model=="gru":
    decoder = GRU(
        n_characters,
        args.hidden_size,
        n_characters,
    )
    corr_fn = utils.get_gru_corrs 

sum_before_outer = False
print("sum_before_outer",sum_before_outer)

decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()

if args.cuda:
    decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

random_ex_fn = functools.partial(random_training_set, args.chunk_len, args.batch_size)

init_corrs = corr_fn(decoder, f'{args.model}_init', random_ex_fn, sum_before_outer=sum_before_outer)

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_ex_fn())
        loss_avg += loss
        all_losses.append(loss)

        if epoch % 5 == 0:
            print('[Time: %s, Epoch: (%d %d%%), Loss: %.4f]' 
                    % (time_since(start), 
                    epoch, epoch / args.n_epochs * 100, 
                    loss)
                )

        if epoch % args.print_every == 0:
            print(generate(decoder, 'Wh', 100, cuda=args.cuda), '\n')

    print("Saving...")
    save()
    end_corrs = corr_fn(decoder, f'{args.model}_end', random_ex_fn, sum_before_outer=sum_before_outer)

    with open(f'results/nfa_corrs_model_{args.model}_lr_{args.learning_rate}.txt', "w") as f:
        f.writelines([str(x) + "," for x in init_corrs])
        f.write(f'\n')
        f.writelines([str(x) + "," for x in end_corrs])

    with open(f'results/train_losses_model_{args.model}_lr_{args.learning_rate}.txt', "w") as f:
        f.writelines([str(x) + "," for x in all_losses])

    with open(f'results/generations_{args.model}_lr_{args.learning_rate}.txt', "w") as f:
        num_gens=10
        for i in range(num_gens):
            f.write("\n")
            f.write(f'Generation {i+1}')
            f.write("\n")
            f.write(generate(decoder, 'Wh', 100, cuda=args.cuda))
            f.write("\n")

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

