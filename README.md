# RoboReg


Pruning (weight removal) work for neural networks
cifar_training notebook for training resnet20 on the cifar10 dataset
cifar_pruning notebook for pruning the resulting neural network.

The idea behind this pruning method:
it looks at the basic building block of resnet (conv1->bn1->conv2->bn2).
For the first layer, the filters are clustered and only the cluster centers are kept.
According to the number of clusters, the weights and biases of bn1 and conv2 are adjusted.
As a result of these changes, the input and output dimensions of the block remain the same, but the number of filters inside the block is reduced, which decreases the number of parameters and consequently the number of flops, speeding up the model inference.
