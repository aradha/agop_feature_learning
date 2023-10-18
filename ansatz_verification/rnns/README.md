This directory contains the code to verify the NFA on RNNs (a basic RNN and a GRU) on two datasets - 
name generation and shakespeare text generation. We modified the implementations from the original tutorials for
Names (https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html) and Shakespeare (https://github.com/spro/char-rnn.pytorch).

For name generation, you will need to create a data folder in the same directory as main.py (for basic RNNs) or gru_main.py (for GRUs). Then, download the data from the Names tutorial to that folder. Call main.py or gru_main.py to verify the NFA. For Shakespeare text generation, you will need to download a file 'shakespeare.txt' from the github tutorial and place this file in the same directory as train.py. To run the NFA verification, run train.py.
