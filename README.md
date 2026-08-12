# Neural-Network
Neural Network from Scratch

A simple feedforward neural network built from scratch in C++ to recognize handwritten digits from the MNIST dataset.

The project focuses on understanding how neural networks actually work internally, without relying on machine-learning frameworks such as TensorFlow or PyTorch.

Architecture
784 input pixels
       go into
128 neurons + ReLU
       narrowed down into
64 neurons + ReLU
       which give us
10 output neurons
       apply the
Softmax function
       and we make a
Digit prediction (0–9)

Each MNIST image is a 28 × 28 grayscale image, flattened into 784 normalized pixel values.

Implemented
Individual neurons with weights and biases
Fully connected neural network layers
Forward propagation
ReLU activation
Softmax output layer
Cross-entropy loss
Backpropagation
Gradient calculation
Gradient descent
Random weight initialization
MNIST binary dataset loader
Basic training loop
Accuracy calculation

Download your own MNIST Files, I am not posting them here due to copyright.
