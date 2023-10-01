import numpy as np
import random
import matplotlib.pyplot as plt

import timeit
import csv
from warnings import warn
from os import path


class Network:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        layers: list,
        testing_data: np.ndarray = None,
        testing_labels: np.ndarray = None,
    ):
        """
        Initialize a Neural Network

        Args:
            data (ndarray): training data for the neural network
            labels (ndarray): training labels for the neural network
            layers (list): layer structure for the neural network
            testing_data (ndarray, optional): testing data for the neural network to predict
            testing_labels (ndarray, optional): testing labels used to confirm predictions
        """
        self.data = (
            data if isinstance(data, np.ndarray) else np.array([], dtype=np.uint8)
        )

        # Onehot encode the labels if possible
        self.labels = np.array([], dtype=np.uint8)

        # Data used for prediction function
        self.testing_data = testing_data
        self.testing_labels = testing_labels

        # Initialize layer sizes, weights, and biases
        self.layers = layers if isinstance(layers, list) else []
        self.weights = []
        self.biases = []

        # If there are given layers, initialize weights and biases
        if len(layers) > 1:
            self._create_w_b(layers)
            self.labels = np.eye(layers[-1])[labels]

        self.accuracies = []

    def _create_w_b(self, layers: list):
        """
        Helper function to create initialize weights and biases

        Args:
            layers (list): list contianing the layer sizes of the network.
        """
        self.weights = [np.array([])] * (len(layers) - 1)
        self.biases = [np.array([])] * (len(layers) - 1)

        # Initialize weights and biases
        for i, item in enumerate(layers):
            if i < len(layers) - 1:
                self.weights[i] = np.random.rand(layers[i + 1], item) - 0.5
                self.biases[i] = np.random.rand(layers[i + 1], 1) - 0.5

    def gradient_descent(
        self,
        training_data: np.ndarray = None,
        training_labels: np.ndarray = None,
        layers: list = None,
        epochs: int = 3,
        learning_rate: float = 0.1,
        rounding: int = 2,
    ):
        """
        Runs the gradient descent algorithm over the training data

        Args:
            training_data (ndarray, optional): data used for predicting. Defaults to None.
            training_labels (ndarray, optional): labels for the training data. Defaults to None.
            layers (list, optional): list contianing the layer sizes of the network. Defaults to None.
            epochs (int, optional): the number of iterations the algorithm runs.
                Defaults to 3.
            learning_rate (float, optional): scaling factor for backpropagation.
                Defaults to 0.1.
            rounding (int, optional): rounding mode for the accuracy recorded. Defaults to 2.
        """

        # Check if all the required data has been loaded into the network
        if isinstance(layers, list):
            self.layers = layers
            if len(layers) > 1:
                self._create_w_b(layers)
        elif layers is not None:
            warn("Invalid layers type provided.")

        if isinstance(training_data, np.ndarray):
            self.data = training_data
            if isinstance(training_labels, np.ndarray):
                self.labels = np.eye(self.layers[-1])[training_labels]
            else:
                warn("Training data and labels may not align properly.")

        # Check for invalid typing
        if not isinstance(epochs, int):
            print("Invalid type provided for epochs: {}".format(type(epochs)))
            return
        if not isinstance(learning_rate, float):
            print(
                "Invalid type provided for learning_rate: {}".format(
                    type(learning_rate)
                )
            )
            return
        if not isinstance(rounding, int):
            print("Invalid type provided for rounding: {}".format(type(rounding)))
            return

        self.accuracies = []
        correct = 0
        for epoch in range(epochs):
            for item, label in zip(self.data, self.labels):
                # Reshaping data and label for
                item.shape += (1,)
                label.shape += (1,)

                # Forward propagation
                node_values = self._forward_prop(item)

                correct += int(np.argmax(node_values[-1]) == np.argmax(label))

                self._backward_prop(node_values, label, learning_rate)
            # Show accuracy for this epoch
            print(f"Epoch: {epoch + 1}")
            print(f"Accuracy: {round((correct / self.data.shape[0]) * 100, rounding)}%")
            self.accuracies.append(
                round((correct / self.data.shape[0]) * 100, rounding)
            )
            correct = 0

    def _forward_prop(self, item):
        """
        Runs the forward propagation algorithm on the given item

        Args:
            item (ndarray): input item to the neural network

        Returns:
            values: node values for each layer in the network
        """
        values = [None] * (len(self.layers) - 1)
        values[0] = self.biases[0] + self.weights[0] @ np.array(item)
        values[0] = 1 / (1 + np.exp(-values[0]))

        for i in range(1, len(self.layers) - 1):
            values[i] = self.biases[i] + self.weights[i] @ np.array(values[i - 1])
            values[i] = 1 / (1 + np.exp(-values[i]))

        return values

    def _backward_prop(self, values: list, label: list, learn_rate: float):
        """
        Runs the backward propagation algorithm for the current items

        Args:
            values (list): list of node values generated from forward propagation
            label (list): one-hot encoded label for the current item
            learn_rate (float): scaling factor for changes in weights and biases
        """
        delta = values[-1] - label
        for i in range(len(self.weights) - 1, 0, -1):
            self.weights[i] += -learn_rate * delta @ values[i - 1].T
            self.biases[i] += -learn_rate * delta
            delta = self.weights[i].T @ delta * (values[i - 1] * (1 - values[i - 1]))

    def save_model(self):
        """
        Saves the model to the models directory
        """
        # Create a new CSV file in the models directory
        # Name schema: model + time_value + accuracy of network
        accuracy = self.accuracies[-1] if len(self.accuracies) > 0 else None
        with open(
            path.join(
                path.dirname(__file__),
                f"models/model{int(timeit.default_timer())}({accuracy}).csv",
            ),
            "w",
            newline="",
        ) as myfile:
            wr = csv.writer(
                myfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            # Write the layers sizes to first line in CSV file
            wr.writerow(self.layers)

            # Write the weights of each node in each layer
            for layer in self.weights:
                for node in layer:
                    wr.writerow(node)

            # Write the biases of each node in each layer
            for layer in self.biases:
                for node in layer:
                    wr.writerow(node)

    def load_model(self, file_path: str):
        """
        Loads the specified model

        Args:
            file_path (str): relative path to the model file
        """
        if not isinstance(file_path, str):
            print("Invalid type provided for file_path: {}".format(type(file_path)))
            return

        data = []
        # Get all rows of the csv file and add them to a list
        with open(file_path, "r") as model_data:
            values = csv.reader(model_data, delimiter=",")
            for row in values:
                data.append(row)

        # Get the layer sizes
        self.layers = [int(layer) for layer in data[:1][0]]

        # Reset weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(self.layers) - 1):
            self.weights.append([])
            self.biases.append([])
            for _ in range(self.layers[1:][i]):
                self.weights[i].append([])
                self.biases[i].append([])

        # Counter for traversing through the model values
        row = 0

        # For each layer after the input layer
        for i in range(1, len(self.layers)):
            # For every node in the layer
            for node in range(self.layers[i]):
                # Load the weights for the node
                self.weights[i - 1][node] = [*map(float, data[node + 1 + row])]
            row += self.layers[i]

        # For each layer after the input layer
        for i in range(1, len(self.layers)):
            # For every node in the layer
            for node in range(self.layers[i]):
                # Load the biases for the node
                self.biases[i - 1][node] = [*map(float, data[node + 1 + row])]
            row += self.layers[i]

    def predict(self, index: int = -1, testing_data=None, testing_labels=None):
        """
        Used to predict a given input

        Args:
            index (int, optional): index in the data to predict. Defaults to -1.
            testing_data (ndarray, optional): data used for predicting. Defaults to None.
            testing_labels (ndarray, optional): labels for the testing data. Defaults to None.
        """
        if index != -1 and not isinstance(index, int):
            print("Invalid type provided for index: {}".format(type(index)))
            return
        if testing_data is not None and not isinstance(testing_data, np.ndarray):
            print(
                "Invalid type provided for testing_data: {}".format(type(testing_data))
            )
            return
        if testing_labels is not None and not isinstance(testing_labels, np.ndarray):
            print(
                "Invalid type provided for testing_labels: {}".format(
                    type(testing_labels)
                )
            )
            return

        # Testing data and labels for prediction
        if testing_data is not None and testing_labels is not None:
            self.testing_data = testing_data
            self.testing_labels = testing_labels

        if (
            self.testing_data is None
            and self.testing_labels is None
            and self.data is None
            and self.labels is None
        ):
            print("Please provide data and labels for the model to predict.")
            return

        # If there is testing data and labels, use them
        if self.testing_data is not None and self.testing_labels is not None:
            values, item = self._predict_helper(
                self.testing_data, index, self.testing_data.shape[0]
            )
        # Otherwise use training data and labels
        elif self.data is not None and self.labels is not None:
            values, item = self._predict_helper(self.data, index, self.data.shape[0])

        plt.imshow(item.reshape(28, 28), cmap="Greys")
        plt.title(values[-1].argmax())
        plt.show()

    def _predict_helper(self, data, index: int, length: int):
        """
        A helper function for the predict function

        Args:
            data (ndarray): array of items the model is trained on.
            index (int): index of item in data.
            length (int): number of items in data

        Returns:
            values (list): node values for each layer in the network
            item (ndarray): item selected from the data
        """
        index = random.randint(0, length) if index == -1 else index
        item = data[index]
        item.shape += (1,)
        values = self._forward_prop(item)
        return values, item

    def show_accuracy(self):
        """
        Displays a plot of the accuracies obtained during gradient descent by epoch.
        """
        if len(self.accuracies) < 2:
            print("Not enough data to display")
            return

        plt.plot(self.accuracies)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.show()
