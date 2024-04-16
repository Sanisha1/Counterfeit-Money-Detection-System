
import numpy as np




class CustomSVM:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_features = X.shape[1]
        self.weights = np.random.uniform(0, 1, num_features)
        self.bias = np.random.uniform(0, 1)

        for epoch in range(self.epochs):
            for features, label in zip(X, y):
                prediction = self.predict(features)
                error = label - prediction

                self.weights += self.learning_rate * error * features
                self.bias += self.learning_rate * error

    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias > 0, 1, 0)