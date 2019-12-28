# Deep Learning

Jeffrey Hinton - Godfather of Deep Learning

- try to mimic how human brain operates
- artificial neural net
  - neurons that receive input values **input layer**
  - **hidden layer** processes input, can have multiple, they receive certain input values
  - neurons that produce output values **output layer**

## How to represent a neuron?

- recreate a *body*, *dendrites* (input), *axon* (output)

In ML:

- Neuron (node) connected to:
  - input layer neurons, each receiving an input value (independent variables for one single observation, like a table row), need to be standardised / normalised
    - or other hidden neurons
  - output signal
    - can be continuous, binary, categorical (multiple outputs)
- Synapses are assigned weights:
  - neural network learn by adjusting weights, by gradient descent and backpropagation

Steps:

1. all weighted sums of input values are summed up in a neuron
2. activation function is applied to neuron value
3. signal is passed on as output (depending on function, might not pass any)

## Activation function

- Threshold function
  - e.g. 1 if x > 0, 0 if x < 0
- Sigmoid function
  - $\frac{1}{1+e^{-x}}$
  - useful to predict probabilities
- Rectifier function
  - max(x,0)
  - increases when x approaches 1
- Hyperbolic tangent (tanh)
  - similar to sigmoid, but goes below 0
  - $\frac{1-e^{-2x}}{1+e^{-2x}}$

Assuming dependent variable is binary:
- can use threshold function (yes/no) or sigmoid function (as probability of yes/no)

Common pattern of rectifier function in hidden layer, sigmoid function at output layer
