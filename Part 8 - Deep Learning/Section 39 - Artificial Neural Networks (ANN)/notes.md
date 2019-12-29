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

## How do neural network learn?

1. perceptron has an output $\hat{y}$ that is compared to actual value $y$
  - cost function: e.g. $C=\frac{1}{2}(\hat{y}-y)^2$
  - goal is to minimise cost function
2. information is fed back to neural network
3. weights get updated
4. repeat

If multiple inputs, cost function is: $C = \sum\frac{1}{2}(\hat{y}-y)^2$ (in batch gradient descent)

## Gradient descent

How to minimise cost function?

**brute force approach**
- impossible with high number of weights
- curse of dimensionality
  - e.g. for 5 inputs variables, have 25 weights, $10^{75}$ combinations
  - fastest supercomputer is Sunway TaihuLight, 93 PFLOPS. 93 x $10^{15}$ operations per second
  - would still require $10^{50}$ years to run optimisation, and that's for a simple network

**gradient descent**

- differentiate, find slope at certain C function, to find minimum
- much faster than brute force

## Stochastic gradient descend

Gradient descend requires convex function, has only one minimum

What if there are multiple local minimums?

- stochastic gradient descend doesn't require for function to be convex
- adjust weights after every row (as opposed to batch gradient descend, which adjust after running all rows)
  - avoid problem of finding local minimum, because it has much higher fluctuation than batch method
  - it's faster than batch
  - rows are picked at random

## Backpropagation

Allows to simultaneously adjust all weights. Algorithm knows how much each weight contributed to error.

# Steps

1. randomly **initialise weights** to small numbers close to 0 (not 0)
2. **input** first observation of dataset in the input layer, each feature in one input node
3. **forward-propagation**: from left to right, neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting predicted results y
4. **compare** predicted result to actual result, measure the generate error
5. **back-propagation**: from right to left, error is back-propagated. Update weights according to how much they are responsible for the error. *Learning rate* decides by how much to update the weights
6. **repeat steps 1 to 5** and update the weights after each observation (reinforcement learning), or after a batch of observations (batch learning)
7. when whole training set has passed through ANN, that makes an epoch. **Redo more epochs**

# Example

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data preprocessing
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1])

labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], 
                        remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:] # avoid the dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create the ANN

# Import Keras and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialise neural network
classifier = Sequential()

# Adding input layer and first hidden layer 
# choose rectifier activation function for hidden layer, sigmoid function for output
# how many hidden nodes? (take average(node in input layer and output)) - units = 6 (11+2)/2
# relu is rectifier
# needs input_dim on first layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform", input_dim = 11))

# add 2nd hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
# if dealing with several categories, will need multiple output neurons, and activation function would be softmax
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
# adam is stochastic gradient descent algorithm
# use logarithmic loss function with sigmoid function (binary_crossentropy with 1 category)
# metric expect analysing parameters in a list, e.g. accuract
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# threshold to change y_pred from probability to true/false
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```