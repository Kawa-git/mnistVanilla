import numpy as np
import time

class NN:
    def __init__(self, numberofneurons, learningrate, epochs) -> None:
        self.numberofneurons = numberofneurons
        self.learningrate = learningrate
        self.epochs = epochs
        
        # Setup size
        input_layer = numberofneurons[0]
        hidden_layer1 = numberofneurons[1]
        hidden_layer2 = numberofneurons[2]
        output_layer = numberofneurons[3]

        # Weights
        self.W1 = np.random.randn(hidden_layer1, input_layer) * np.sqrt(1.0/hidden_layer1)  #from input -> hidden1, size=128x784
        self.W2 = np.random.randn(hidden_layer2, hidden_layer1) * np.sqrt(1.0/hidden_layer2)    #from hidden1 -> hidden2, size=64x128
        self.W3 = np.random.randn(output_layer, hidden_layer2) * np.sqrt(1.0/output_layer)  #from hidden2 -> output, size=10x64

    def sigmoid(self, x) -> float:
        return 1/(1+np.exp(-x))

    def sigmoid_d(self, x) -> float:
        return (np.exp(-x))/((np.exp(-x)+1)**2)

    def softmax(self, x) -> float:
        return ((np.exp(x-x.max())) / np.sum((np.exp(x-x.max())), axis=0))

    def softmax_d(self, x) -> float:
        return (np.exp(x-x.max())) / (np.sum(np.exp(x-x.max()), axis=0) * (1-(np.exp(x-x.max()))/np.sum((np.exp(x-x.max())), axis=0))) 

    def forward_propagation(self, x_train) -> float:
        self.A0 = x_train   #784x1

        # Input to first hidden layer
        self.Z1 = np.dot(self.W1, self.A0)  #128x1
        self.A1 = self.sigmoid(self.Z1)

        # First hidden layer to second hidden layer
        self.Z2 = np.dot(self.W2, self.A1)
        self.A2 = self.sigmoid(self.Z2) 

        # Second hidden layer to output layer
        self.Z3 = np.dot(self.W3, self.A2)
        self.A3 = self.softmax(self.Z3)
        
        return self.Z3

    def backward_propagation(self, y_train, output):
        weight_change = {}

        # Update the third set of weights
        error = 2 * (output - y_train) / output.shape[0] * self.softmax_d(self.Z3)
        weight_change["W3"] = np.outer(error, self.A2)

        # Update the second set of weights
        error = np.dot(self.W3.T, error) * self.sigmoid_d(self.Z2)
        weight_change["W2"] = np.outer(error, self.A1)

        # Update the first set of weights
        error = np.dot(self.W2.T, error) * self.sigmoid_d(self.Z1)
        weight_change["W1"] = np.outer(error, self.A0)

        return weight_change

    def update_weights(self, weight_change):
        self.W1 -= self.learningrate * weight_change["W1"]
        self.W2 -= self.learningrate * weight_change["W2"]
        self.W3 -= self.learningrate * weight_change["W3"] 

    def compute_accuracy(self, test_dataset) -> None:
        predictions = []
        for x in test_dataset:
            values = x.split(",")
            inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            targets =  np.zeros(10) + 0.01
            targets[int(values[0])] = 0.99
            output = self.forward_propagation(inputs) 
            pred = np.argmax(output)
            predictions.append(pred==np.argmax(targets))
        return np.mean(predictions)

    def train(self, train_dataset, test_dataset) -> None:
        for i in range(self.epochs):
            start_time = time.time()
            for x in train_dataset:
                values = x.split(",")
                inputs = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                targets =  np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99
                output = self.forward_propagation(inputs) 
                change_w = self.backward_propagation(targets, output)
                self.update_weights(change_w)
            accuracy = self.compute_accuracy(test_dataset)
            print("Epoch: {0}, Time Spent: {1:.02f}s, Accuracy: {2:.02f}%".format(i+1, time.time() - start_time, accuracy*100))


with open("dataset/mnist_train.csv", "r") as train_dataset:
    train_data = train_dataset.readlines()

with open("dataset/mnist_test.csv", "r") as train_dataset:
    test_data = train_dataset.readlines()
    
# 784 input nodes, 128 hidden nodes, 64 hidden nodes, 10 output nodes
nn = NN(numberofneurons=[784, 128, 64, 10], learningrate=0.01,epochs=10) 
nn.train(train_data, test_data)  



