Code for verifying the Neural Feature Ansatz for residual networks and VGG networks pre-trained on ImageNet.  This code requires download of the ImageNet dataset in advance (`https://www.image-net.org/download.php`).  The path to ImageNet needs to provided in line 309 for `verify_resnet_ansatz.py` and in line 199 of `verify_vgg_ansatz.py`.

In `verify_resnet_ansatz.py`, note that lines 159 and 178-179 need to be commented out and lines 161 and 182-183 need to be uncommented in order to use architectures resnet50, resnet101, resnet152.  This is due to these models using `Bottleneck` layers instead of the `BasicBlock` layers used in resnet18, resnet34. 