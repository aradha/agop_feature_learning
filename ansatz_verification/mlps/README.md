Code for verifying Neural Feature Ansatz in MLPs.

# Training Neural Networks
Code for training MLPs on datasets such as SVHN, CelebA, etc. is available in main.py.  This code will attempt to save trained models to a directory named `saved_nns`, which should be created before running the code.  

# Verifying Ansatz
Code for verifying the ansatz is available in `verify_deep_NFA.py`.  It will require you to load a saved neural net and a corresponding dataset.  

# Software
Software versions are available via the file deep_nfa_env.yml.  This code primarily requires pytorch version 1.13 with functorch (installed via pip).  
