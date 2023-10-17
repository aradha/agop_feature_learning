Code for verifying Deep Neural Feature Ansatz from https://arxiv.org/abs/2212.13881

# Training Neural Networks
Code for fully connected networks on all datasets considered in the paper is available in main.py.  This code will try to save neural networks to a directory named saved_nns.  

# Verifying Ansatz
Code for verifying the ansatz is available in verify_deep_NFA.py.  It will require you to load a saved neural net and a corresponding dataset.  

# Software
Software versions are available via the file deep_nfa_env.yml.  This code primarily requires pytorch version 1.13 with functorch (installed via pip).  
