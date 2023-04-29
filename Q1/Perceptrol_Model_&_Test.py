# Importing Libraries

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

"""
# MyPerceptron Class
"""
class MyPerceptron:

    def __init__(self, learning_rate=1, n_iterations=3):

        self.lr = learning_rate  # Default learning rate is 0.1
        self.epochs = n_iterations  # epoch is for number of data set to iterate
        self.weights = None

    def fit(self, X, y):
        """
        Target is to calculate the final weight by using below equation:
        New_Weights = Old_Weights + learning_rate *(Expected_Y[i] - Predicted_Y) * input_x[i])
                                    or
        weights(t + 1) = weights(t) + learning_rate * (expected_i – predicted_) * input_i
        """

        # print("X shape", X.shape)
        # print("y shape", y.shape)

        # Default weights value set as [0. 0.]
        self.weights = np.zeros(X.shape[1])

        # Set given initial weights value as [1. 1.]
        # self.weights = np.ones(X.shape[1])

        #print(self.weights)


        for epoch in range(self.epochs):
            for i in range(X.shape[0]):

                y_pred = self.activation_function(np.dot(self.weights, X[i]))

                #self.weights = self.weights([w1, w2]) + self.lr * (y[i] - y_pred) * X[i]([x1, x2])
                self.weights = self.weights + self.lr * (y[i] - y_pred) * X[i]

                print(f"{i} times Weight Vectors: ", self.weights, " & Predicted Value: ", y_pred)

            print(f"\n{epoch} Epochs\Iterations Weight Vectors: ", self.weights, " & Predicted Value: ", y_pred, "\n")

        print("Training Complete")
        print(f"Number of steps ({epoch+1} * {i+1}) : ", (epoch + 1) * (i + 1))

        #print(self.weights)

    # Activation function will use for y_pred
    def activation_function(self, activation):

        if activation >= 0:
            return 1
        else:
            return -1

    # Predict function for given input X
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.activation_function(np.dot(self.weights, X[i])))
        return np.array(y_pred)

"""
# Decision Boundary Function to plot the scatter diagram
"""
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    #print(Z)
    Z = Z.reshape(xx1.shape)
    print(Z)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


"""
# Sample Data Set:
X = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
# print(X)
y = np.array([1, -1, -1, -1, 1, 1])
# print(y)
"""

data = pd.read_csv("Perceptron_Sample_Data.csv")
print("\nSample Data Set:\n", data, "\n")
print("Below Iterate weights for each epochs: \n")

# x1 & X2 weights value
X = data.iloc[:, 0:-1].values
# print(X)

# Class value
y = data.iloc[:, -1].values
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


model = MyPerceptron()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nFinal weight vector: ", model.weights)
print("Final Predicted y value: ", y_pred)

print("\nAccuracy Score is: ", accuracy_score(y_test, y_pred))

# calling Decision Boundary Function to plot the scatter diagram
print("\nDecision Boundary Predicted value:")
plot_decision_boundary(X_train, y_train, model)
