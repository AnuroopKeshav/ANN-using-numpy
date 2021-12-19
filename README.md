# ANN Model using Numpy
A simple Artificial Neural Network tool built using Numpy library that can be implemented to build a classification model with a desired architecture.

## Parameters
The parameters of the model include

* #### learning_rate:
    * Defines the rate at which the model learns. A high value for this can cause overshooting, causing the model to be inaccurate and a low value could result in smaller steps towards minima, meaning slow convergence and requiring more iterations.
    * Default: `0.1`

* #### epochs:
    * This parameter defines the number of times the model sees the whole dataset during the training process.
    * Default: `10`

The parameters of the layers include

* #### n_perceptrons:
    * This value defines the number of perceptrons (neurons) in the particular hidden layer. While most of choosing an appropriate number comes down to trial and error and intuition, there are few methods that give a rough approximation for the optimum number of perceptrons in a layer. Also, there are two tips that could deem helpful in most cases.

        * The number of perceptrons in the hidden layer should be between the size of the input and the output perceptrons.
        * Usage of `GridSearchCV` or `RandomSearchCV` helps find the suitable number of perceptrons for the case.

* #### activation_function:
    * The activation function squishes the value obtained by the linear combination of the inputs (previous layer outputs) and weights + bias and maps it to a value in a specific range.
    * Default: `tanh`

## Activation Function List
* `relu`
* `leakyrelu` (p=0.1)
* `tanh`
* `sigmoid`
* `softmax`

## A generic architecture of an ANN model
An ANN model consists of three types of layers.

### Input Layer
This layer is where the input i.e., the dataset is passed. Therefore, the number of perceptrons in this layer will be equal to the number of features or independent variables in the dataset.

### Hidden Layers
The hidden layers are where the "magic" happen. A series of operations take place here ultimately leading to the output layer. More about it in the [intuition](#Intuition) section.

### Output Layer
Output layer is where the chain of operations come to an end and the outputs are either compared to the actual results (in training process) or used as the predictions.

<img src="https://1.cms.s81c.com/sites/default/files/2021-01-06/ICLH_Diagram_Batch_01_03-DeepNeuralNetwork-WHITEBG.png">

## Intuition
A brief explanation of the working of the model is elaborated below. However, watching [3Blue1Brown's playlist on Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) is highly suggested as it is very insightful.

The learning process of ANN model comprises of two main steps.

### Feed Forward
* This process is where all the features of input are multiplied with the corresponding randomised weights and added together to the bias to get a value. The obtained value is further squished into yet another value within a specific range using activation function.

* Let us first consider a single perceptron in the first hidden layer.

	<img src="https://latex.codecogs.com/svg.latex?\Large&space;a_{1}=\sigma_{0}(w_{0}x_{0}+w_{1}x_{1}+w_{2}x_{2}+...+w_{n}x_{n}+b_{0})"/>

* Also expressed as

	<img src="https://latex.codecogs.com/svg.latex?\Large&space;a_{1}=\sigma_{0}(z_{0}=(\sum_{i=0}^{n}w_{i}x_{i}+b_{0}))"/>

* Where,

    * a₁ is the output value of the first hidden layer (single perceptron).
    * xᵢ is the feature of the dataset.
    * wᵢ is the weight corresponding to xᵢ.
    * b₀ is the bias of the first hidden layer (single perceptron).
    * z₀ is the sum of the linear combination and bias corresponding to the first layer.
    * σ₀ is the activation function of the first hidden layer that takes in z₀ as input.

* Since there are multiple perceptrons in the hidden layers, we use matrix to get all the values simultaneously.

	<img src="https://latex.codecogs.com/svg.latex?\Large&space;A_{1[m\times&space;n_{1}]}=(\sigma_{0}(W_{0[n_{1}\times&space;n_{0}]}\cdot&space;X_{[m\times&space;n_{0}]}^{T}%20+%20B_{0[n_{1}]}))^{T}"/>

* Where,

    * A₁ is the output matrix of the first hidden layer and the input to the second hidden layer.
    * X is the input matrix.
    * W₀ is the weight matrix of the first hidden layer.
    * B₀ is the bias matrix for every perceptron in the first hidden layer
    * m is the number of rows/entries in the dataset.
    * n₀ is the number of features in the input dataset.
    * n₁ is the number of perceptrons in the first hidden layer.
    * σ₀ is the activation function of the first hidden layer.

* The process continues with A₁ being the input for the next hidden layer to obtain A₂ and so on until the last layer returns an output matrix.

### Back Propagation
* The output matrix of the model is then compared to the [One Hot Encoded](https://www.geeksforgeeks.org/ml-one-hot-encoding-of-datasets-in-python/) version of the actual resultant matrix.

* Average error for the last layer is calculated using the cost function

	<img src="https://latex.codecogs.com/svg.latex?\Large&space;C.F=\frac{1}{2m}\sum_{i=0}^{m}(a_{pi}-y_{i})^2"/>
	(single perceptron version)

* Where,

    * C.F is the cost function for the output layer.
    * aₚ is the output of the last layer, i.e., the output layer.
    * yᵢ is the actual output.
    * m is the number of entries in the dataset.

* The purpose of 2 in the denominator is simply to cancel out the coefficient obtained when we differentiate the function in further steps.

* The matrix interpretation would be

	<img src="https://latex.codecogs.com/svg.latex?\Large&space;C.F=(A_{p[m\times&space;n_{p}]}-Y_{[m\times&space;n_{p}]})^2"/>

* Where,

    * C.F is the cost function for the output layer.
    * Aₚ is the output matrix of the last layer, i.e., the output layer.
    * Y is the resultant matrix.
    * m is the number of entries in the dataset.
    * nₚ is the number of perceptrons in the output layer.

* Provided we start off from a random set of weights and biases, we should now have a random set of results and thus some error. In order to reduce the error (reach the minima of the cost function), we must alter the weight and bias matrix.

#### Gradient Descent

* To update the weight matrix, we first find the slopes (derivatives) of the cost function with respect to the weights and bias.

    * First we differentiate the cost function with respect to the weights and biases of the final layer and keep the derivatives aside.
    * Then we differentiate the cost function with respect to the input for that layer, which infact is the output of the previous layer. By doing this we will now be able to differentiate the cost function with respect to the weights and biases of the previous layer. The chain rule continues until we reach the first layer and have the derivatives with respect to all the weights and biases.

* Once this is done, a simple subtraction from the existing corresponding set of weights and biases should minimize the output of the cost function, hence the minimizing the error.
* All we are essentially doing is going along the direction of the slope in the GIF shown below.

![](https://cdn-images-1.medium.com/max/720/1*lhEF_VbpXHW76p6KI5cycQ.gif)

* Only that there are generally much more axes for weights and biases than the above image where there are two axes.
* After significant number of iterations, the value of the cost function should have neared the minimum.
* This process is only briefly explained as I believe [3Blue1Brown's video on back propagation](https://www.youtube.com/watch?v=tIeHLnjs5U8) does a better job explaining this part thoroughly.

## Usage

```py
    from model import Model, Layer

    # Building the model
    model = Model(learning_rate=0.065, epochs=20)
    model.add(Layer(64, activation_function='relu'))
    model.add(Layer(64, activation_function='relu'))
    model.add(Layer(10, activation_function='softmax'))

    # Fitting the model with the dataset
    model.fit(X_train, y_train)

    # Predicting using the fit model
    model.predict(X_test)
```
