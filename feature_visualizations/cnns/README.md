Code for visualizing AGOP features for CNNs pre-trained on ImageNet.  An example is provided for VGG19.  This code requires download of the ImageNet dataset in advance (`https://www.image-net.org/download.php`).  The path to ImageNet needs to provided in line 107 of `vgg_feature_vis.py`. 

If AGOPs are provided by setting the path to the corresponding AGOP file in line 106, then the code will visualize the features extracted by AGOP on a given layer indexed by `layer_idx`.  Otherwise, the code will default to using the Neural Feature Matrix for visualization. 

AGOPs for VGG19 are provided via the following link [VGG19-AGOPs](https://drive.google.com/drive/u/1/folders/1d75clBAX58Vn4m2Gv7ymhMSG1QN8O4wG).
