# RoboReg

The pruning technique applied in this neural network approach involves removing weights in a ResNet20 model, trained on the CIFAR-10 dataset. The cifar_training notebook is used to train the ResNet20 on the dataset, while the cifar_pruning notebook is utilized to prune the trained network.

The concept behind this pruning method focuses on the core structure of ResNet, consisting of a sequence: conv1 -> bn1 -> conv2 -> bn2. In the first convolutional layer, the filters are grouped into clusters, and only the cluster centers are retained. Based on the number of clusters, the weights and biases of the subsequent bn1 and conv2 layers are adjusted. This process keeps the input and output dimensions of the block unchanged, but reduces the number of filters inside it. As a result, the number of parameters is decreased, leading to a reduction in floating-point operations (FLOPs), which accelerates model inference..
