# Neural-Networks ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This is based off of the presentation by Bot Academy. You can find the code [here](https://github.com/Bot-Academy/NeuralNetworkFromScratch/blob/master/nn.py) and the video [here](https://youtu.be/9RN2Wr8xvro).

This is an educational project that I am working on, while learning about Neural Networks and machine learning.

Currently there is a class called Network implemented in the Network.py file. The version implemented in the video by Bot Academy was a hard-coded neural network with a hidden layer of 20 nodes, input layer of 784 nodes, and output layer of 10 nodes. The network I have implemented is a generic network, where any layers sizes are accepted, and any networks that have at least an input layer and an output layer are accepted.

My intent with this project is to create an object oriented neural network that is essentially plug-and-play.

## Examples:

To use the neural network, you need training data, labels for the training data, and a list of the layers' node counts.

```
# Loading MNIST dataset from tensorflow
(x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")
x_train = np.reshape(x_train, (x_train.shape[0], 784), order='C')

network = Network(x_train, y_train, [784, 100, 10])
```

The gradient descent function is then called on the network to train it. This takes in six optional arguments: ```training_data``` (ndarray: data used for predicting), ```training_labels``` (ndarray: labels for the testing data.), ```layers``` (list: list contianing the layer sizes of the network), ```epochs``` (int: the number of iterations the algorithm runs), ```learning_rate``` (float: scaling factor for backpropagation), ```rounding``` (int: rounding mode for the accuracy recorded.). 

The default values are: ```training_data``` = None, ```training_labels``` = None, ```layers``` = None, ```epochs``` = 3, ```learning_rate``` = 0.1, ```rounding``` = 2.

```
network.gradient_descent()
```

## Save & Loading

There are two functions implemented to save and load a model, respectively.

The ```save_model()``` function will save the layer structure, weights, and biases for the network in a CSV file.

Output file format:
```
Line 1: [Layer structure]
Line 2->n: [weights]
Line n+1->m: [biases]
```

For the file, the ```n``` is the sum of the number of layers in the network, excluding the input nodes. The value ```m``` is two times the value of ```n```.

The ```load_model()``` function take a required parameter ```file_path``` which is the relative path to the model file. This will set the layers, weights, and biases for the network. However, training data is still needed (for now) to use the predict function.

In the ```/models``` directory, there is an example model file. It is a network trained for the MNIST dataset. To use the model, create your ```Network``` object, like is done above in ```Examples```, then: ```network.load_model("models/Model-Example.csv")```. You can then use the testing function to predict images from the given training data.

## Testing

To test the network, there is a prediction function implemented: ```predict()```. It takes an optional argument ```index``` for the data point from the dataset to be tested against. If no index is given, a random datum is chosen to be tested against. You can also provide ```testing_data``` and ```testing_labels``` for the model to predict; these parameters require type ```ndarray```.
