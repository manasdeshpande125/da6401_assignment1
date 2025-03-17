# DA6401 Assignment 1

## Overview
This repository contains an implementation of a fully connected **neural network** from scratch in Python. The network supports multiple hidden layers, various optimizers, and activation functions. It can be trained and evaluated on either the **Fashion-MNIST** or **MNIST** datasets.

## Features
- Supports **multiple hidden layers** with customizable sizes.
- Implements **SGD, Momentum, NAG, RMSprop, Adam, and Nadam** optimizers.
- Supports **ReLU, Sigmoid, Tanh, and Identity** activation functions.
- Supports **Random and Xavier** weight initialization.
- Allows **L2 regularization (Weight Decay)** to prevent overfitting.
- Trains on **Fashion-MNIST** or **MNIST** datasets.

---

## Installation
Ensure you have Python 3 installed, then install dependencies:
```bash
pip install numpy matplotlib wandb
```
Also I have imported datasets from keras.datasets

---

## Training the Model
To train the model, use the `train` function with the following parameters:

### **Function Signature:**
```python
train(inputSize, hiddenLayers, outputSize, sizeOfHiddenLayers, batchSize, learningRate,
      initialisationType, optimiser, epochs, activationFunc, weightDecay, lossFunc,
      dataset, beta, beta1, beta2, epsilon)
```

### **Parameters:**
| Parameter             | Description |
|----------------------|-------------|
| `inputSize`         | Number of input features |
| `hiddenLayers`      | Number of hidden layers |
| `outputSize`        | Number of output classes |
| `sizeOfHiddenLayers`| List containing the number of neurons per hidden layer |
| `batchSize`         | Number of samples per batch |
| `learningRate`      | Learning rate (e.g., `0.001`) |
| `initialisationType`| Weight initialization (`"random"` or `"xavier"`) |
| `optimiser`         | Optimization algorithm (see below) |
| `epochs`            | Number of training epochs |
| `activationFunc`    | Activation function (see below) |
| `weightDecay`       | L2 regularization strength |
| `lossFunc`          | Loss function (e.g., `"cross_entropy"`) |
| `dataset`           | Either `"fashion_mnist"` or `"mnist"` |
| `beta`              | Momentum term (for `momentum`, `NAG`, etc.) |
| `beta1` & `beta2`   | Adam/Nadam specific parameters |
| `epsilon`           | Small constant for numerical stability |

### **Example Usage:**
```python
train(inputSize=784, hiddenLayers=4, outputSize=10, sizeOfHiddenLayers=64,
      batchSize=32, learningRate=0.001, initialisationType="xavier", optimiser="nadam",
      epochs=10, activationFunc="relu", weightDecay=0.0001, lossFunc="cross_entropy",
      dataset="fashion_mnist", beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8)
```

---

## Optimizers
The model supports the following optimization algorithms:
- **SGD** (Stochastic Gradient Descent)
- **Momentum-Based Gradient Descent**
- **Nesterov Accelerated Gradient Descent (NAG)**
- **RMSprop**
- **Adam**
- **Nadam** (Nesterov-accelerated Adam)

---

## Activation Functions
Supported activation functions:
- `tanh`
- `sigmoid`
- `relu`
- `identity`

---

## Weight Initializers
Supported weight initialization methods:
- `random`
- `xavier`



---

## Best Observations
From experimentation, the best configuration achieved **88.23% validation accuracy**:
- **Epochs:** 10
- **Hidden Layers:** 4 (each with 64 neurons)
- **Learning Rate:** 0.001
- **Optimizer:** aadam
- **Batch Size:** 32
- **Weight Initialization:** Xavier
- **Activation Function:** ReLU
- **Loss Function:** Cross-Entropy

For better performance, **data augmentation** can be used to reach **95% accuracy**.

---
## Wandb Report Link:
https://api.wandb.ai/links/manasdeshpande4902-iit-madras/hxjbuexn

---
## Way to run train.py
python train.py  -we "manasdeshpande4902-iit-madras" -wp "Trial" -o "adam" -lr 0.001 -l "cross_entropy" -e 10 -w_i "xavier" -w_d 0.005 -a "tanh"


