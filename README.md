# RoboReg


#CIFAR-10 Pruning with ResNet-20

Summary:

This project implements network pruning on a ResNet model using the CIFAR-10 dataset. The goal is to prune the network to reduce its size and inference time while maintaining accuracy. The project leverages data augmentation and pruning techniques to optimize the deep learning model.
The pruning technique applied in this neural network approach involves removing weights in a ResNet20 model, trained on the CIFAR-10 dataset. The cifar_training notebook is used to train the ResNet20 on the dataset, while the cifar_pruning notebook is utilized to prune the trained network.

The concept behind this pruning method focuses on the core structure of ResNet, consisting of a sequence: conv1 -> bn1 -> conv2 -> bn2. In the first convolutional layer, the filters are grouped into clusters, and only the cluster centers are retained. Based on the number of clusters, the weights and biases of the subsequent bn1 and conv2 layers are adjusted. This process keeps the input and output dimensions of the block unchanged, but reduces the number of filters inside it. As a result, the number of parameters is decreased, leading to a reduction in floating-point operations (FLOPs), which accelerates model inference..

## Dataset

The **CIFAR-10 dataset** is used for training and testing the model. It consists of 60,000 32x32 color images in 10 different classes with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 testing images.

### Data Augmentation:

During training, the following augmentations are applied to the dataset:

- **Random Cropping** with padding.
- **Random Horizontal Flip**.
- **Random Rotation** of up to 10 degrees.
- **Random Affine Transformation** with shear and scaling.
- **Color Jitter** for brightness, contrast, and saturation.
- **Normalization** to mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

## Model Architecture

The model is a **ResNet** (Residual Network) implemented using PyTorch. Key elements include:

- **BasicBlock**: Consists of two convolutional layers with batch normalization and ReLU activation. A **shortcut connection** is applied for residual learning.
- **LambdaLayer**: Helps manage shortcuts when input/output shapes differ.

The network is composed of:

- Three convolutional layers with batch normalization.
- A final fully connected layer for class prediction.
- The network has approximately 1.7 million parameters before pruning.

## Training Methodology

The model is trained using:

- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Stochastic Gradient Descent (SGD) with momentum of `0.9` and weight decay of `1e-4`.
- **Learning Rate Scheduler**: MultiStepLR scheduler with milestones at epochs 100 and 150.

### Training Steps:

- **Training Phase**: The model's predictions are compared with ground truth, and loss is backpropagated to update weights.
- **Validation Phase**: Accuracy is computed by comparing the model’s predictions against the validation set.

## Pruning Strategies

**Network Pruning** reduces the size of the model by removing unnecessary parameters (filters, neurons, or entire layers) while preserving as much performance as possible.

### Techniques Used:

1. **Weight Pruning**: Small weight connections are pruned based on a magnitude threshold.
2. **Filter Pruning**: Certain filters in convolutional layers are removed if deemed unnecessary.

Pruning helps to:

- **Reduce Inference Time**: The pruned model runs faster.
- **Reduce Model Size**: The model size is reduced for deployment on edge devices.

After pruning, the model is fine-tuned to recover some of the lost performance.


## Results

The model achieved the following precision, recall, and F1 scores on the CIFAR-10 test dataset:

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Plane    | 0.80      | 0.84   | 0.82     | 1000    |
| Car      | 0.92      | 0.85   | 0.88     | 1000    |
| Bird     | 0.82      | 0.77   | 0.80     | 1000    |
| Cat      | 0.65      | 0.75   | 0.70     | 1000    |
| Deer     | 0.83      | 0.86   | 0.85     | 1000    |
| Dog      | 0.84      | 0.74   | 0.78     | 1000    |
| Frog     | 0.89      | 0.84   | 0.86     | 1000    |
| Horse    | 0.83      | 0.91   | 0.87     | 1000    |
| Ship     | 0.89      | 0.88   | 0.88     | 1000    |
| Truck    | 0.87      | 0.85   | 0.86     | 1000    |

- **Overall Accuracy**: 83%
- **Macro Avg Precision**: 83%
- **Macro Avg Recall**: 83%
- **Macro Avg F1-Score**: 83%

## Training and Pruning Accuracy Comparison

1. **Before Pruning** (Initial Model):
   - **Accuracy**: **82.91%**
   - **Number of Parameters**: 269,722

2. **After Pruning**:
   - **k = 2 (Aggressive Pruning)**:
     - **Accuracy**: **10.04%**
     - **Number of Parameters**: 13,054
   - **k = [8, 8, 8, 16, 16, 16, 32, 32, 32] (Moderate Pruning)**:
     - **Accuracy**: **51.28%**
     - **Number of Parameters**: 135,754
   - **k = [15, 15, 15, 31, 31, 31, 63, 63, 63] (Minimal Pruning)**:
     - **Accuracy**: **82.37%**
     - **Number of Parameters**: 264,088

### Insights:
- The **accuracy dropped** slightly after minimal pruning (from 82.91% to 82.37%).
- However, the **number of parameters** was reduced, demonstrating that the model became slightly more efficient with almost the same accuracy.

---

## Correlation Analysis

The correlation between accuracy and the number of parameters was assessed using both **Pearson** and **Spearman** methods.

### Pearson Correlation:

| Metric           | `acc` (Accuracy) | `num_parameters` (Number of Parameters) |
|------------------|------------------|-----------------------------------------|
| **i (Layer i)**  | 0.319270          | 0.066326                                |
| **j (Layer j)**  | 0.314795          | 0.242432                                |
| **k (Layer k)**  | 0.868530          | 0.967899                                |
| **acc**          | 1.000000          | 0.938141                                |
| **num_parameters**| 0.938141         | 1.000000                                |

### Spearman Correlation:

| Metric           | `acc` (Accuracy) | `num_parameters` (Number of Parameters) |
|------------------|------------------|-----------------------------------------|
| **i (Layer i)**  | 0.361076          | 0.104828                                |
| **j (Layer j)**  | 0.308662          | 0.314485                                |
| **k (Layer k)**  | 0.856099          | 0.943456                                |
| **acc**          | 1.000000          | 0.942613                                |
| **num_parameters**| 0.942613         | 1.000000                                |

### Insights:
- **Layer k** had the strongest correlation with accuracy and the number of parameters, indicating its sensitivity to pruning.
- Layers **i** and **j** showed weaker correlations, suggesting pruning these layers is less impactful on overall performance.

