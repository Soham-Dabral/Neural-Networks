# Neural Network from Scratch (v1)

A simple feed-forward neural network implemented entirely from scratch using pure Python.

This project was created to understand the mathematics and mechanics behind neural networks before using professional machine learning frameworks such as PyTorch.

No machine learning libraries are used.

---

## Features

* Feed-forward neural network
* Object-oriented implementation
* Multiple inputs
* Multiple outputs
* Multiple neurons per layer
* Forward propagation
* Backpropagation
* Gradient Descent optimization
* Mean Squared Error (MSE) loss
* Configurable learning rate

---

## Project Structure

```text
NeuralNetwork
│
├── Neuron
│   ├── forward()
│   └── backward()
│
├── Layer
│   ├── forward()
│   └── backward()
│
└── NeuralNetwork
    ├── forward()
    ├── backward()
    └── calculate_loss()
```

---

## How it Works

### Forward Propagation

Each neuron computes

$$
y = \sum_i w_i x_i + b
$$


The outputs from one layer become the inputs of the next layer.

---

### Backpropagation

Gradients are propagated backward through the network.

For every neuron:

* Weight gradients are computed.
* Bias gradients are computed.
* Gradients with respect to the inputs are returned to the previous layer.

Weights and biases are updated using Gradient Descent.

---

## Training

The network is trained using supervised learning.

Training loop:

1. Forward pass
2. Compute loss
3. Compute output gradients
4. Backpropagate gradients
5. Update parameters

Repeat for multiple epochs.

---

## Example Output

```text
Epoch Loss:
1.6275186550841e-28

Input      Prediction                 Expected

[0,0]   -> [1.0000000000000053, 2.0]  [1,2]
[1,0]   -> [3.000000000000004 , 1.0]  [3,1]
[0,1]   -> [2.000000000000003 , 5.0]  [2,5]
...
```

The network successfully learns linear mappings with prediction errors approaching floating-point precision.

---

## Limitations

This is an educational implementation.

Current limitations include:

* Linear neurons only
* No activation functions
* No automatic network generation
* No batching
* No optimizers beyond Gradient Descent
* No GPU acceleration
* No automatic differentiation

---

## Future Improvements

Planned features include:

* Automatic network builder
* Activation functions (ReLU, Sigmoid, Tanh)
* Softmax output
* Cross-Entropy loss
* Xavier/He initialization
* Additional optimizers
* NumPy implementation
* PyTorch implementation
* Integration into the TWT Chatbot project

---

## Purpose

The objective of this project is not to compete with existing machine learning frameworks.

Instead, it serves as a learning exercise to understand the internal workings of neural networks before transitioning to professional frameworks such as PyTorch.

---

## License

This project is intended for educational purposes.
