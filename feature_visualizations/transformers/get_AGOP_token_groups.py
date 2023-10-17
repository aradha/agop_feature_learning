import os
import torch
from numpy.linalg import svd
from numpy.linalg import norm as np_norm
import numpy as np
from torch.linalg import norm
import random
from transformers import GPT2TokenizerFast
import hickle
import visdom


SEED = 1717

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

def agop_svd(agops, decode, fnames, outf):

    # EXAMPLES OF THEMATIC SINGULAR VECTORS
    #keys = [(8, 2, 0)] # food
    #keys = [(5, 5, 2)] # adjectives
    #keys = [(0, 1, 8)] # past-tense verbs
    keys = [(2, 1, 0)] # names

    for agop_idx, MV in enumerate(agops):
        U, s, Vt = svd(MV)
        layer_head = fnames[agop_idx].strip().split("_")
        head = eval(layer_head[-3])
        layer = eval(layer_head[-1][:-2])

        for j in range(len(Vt)):
            if (j, head, layer) not in keys:
                continue
            V1 = Vt[j,:].reshape(-1, 1)
            return np.concatenate([V1], axis=-1)


def get_bpe(BPE_PATH):
    enc = GPT2TokenizerFast.from_pretrained(BPE_PATH)

    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)
    return encode, decode

def load_agops(AGOP_DIR):
    fnames = sorted([AGOP_DIR + f for f in os.listdir(AGOP_DIR)], key=lambda x: x.split('_')[-1])
    agops = []
    for fname in fnames:
        GV = hickle.load(fname)
        agops.append(GV)
    return agops, fnames


def decode_example(example, decode):
    return list([decode(e) for e in example])


def main():
    BPE_PATH = '/home/aradha/NeuralModels/transformers/nanogpt/tokenizers/3200_vocab_bpe/'
    TOKEN_EMBEDDING_PATH = 'token_embedding.h'
    AGOP_DIR = '/home/aradha/NeuralModels/transformers/nanogpt/saved_agops/'

    agops, fnames = load_agops(AGOP_DIR)
    encode, decode = get_bpe(BPE_PATH)

    E = hickle.load(TOKEN_EMBEDDING_PATH)
    Vs = agop_svd(agops, decode, fnames, None)

    # Customize text as desired
    text = "Anna and Ben are friends. They like to play games. Today they play a game with cards. \"Who will win?\" Anna asks. \"I will win,\" Ben says."

    ix = encode(text)
    w = E[ix, :] @ Vs
    pairs = [(decode_example(ix, decode), ' ', list(w.reshape(-1)))]

    hickle.dump(pairs, 'token_weight_pairs.h')


if __name__ == "__main__":
    main()
