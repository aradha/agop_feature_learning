Code for verifying the Neural Feature Ansatz for vision transformers networks pre-trained on ImageNet. This code requires download of the ImageNet dataset in advance (https://www.image-net.org/download.php). The path to ImageNet needs to provided in line 318 of `verify_vis_transformer_ansatz.py`.  Additionally, the filename in line 315 needs to be set for logging query, key, value correlations. 

The code is currently set to compute AGOP with respect to 10 images from ImageNet. The number of samples used in AGOP can be modified by updating the `MAX_IDX` variable in line 232.  
