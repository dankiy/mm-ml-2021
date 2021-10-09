import numpy as np

from ..abstract import Model


class LinRegSGD(Model):
    """
    Linear regression model with stochastic gradient descent

    Args:
        l1_reg (float): L1 regularization strength
        l2_reg (float): L2 regularization strength
        max_loss (float): loss limitation
        random_seed (int): value for numpy random number generator initialization

    Methods:
        fit (X_train, y_train, learning_rate=1e-4, epochs=100): perform model learning
                    for training dataset
        predict (X): perform model prediction for array of samples

    """
    def __init__(
        self,
        l1_reg: float = 0.,
        l2_reg: float = 0.,
        max_loss: float = 1e12,
        random_seed: int = 0
    ):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.max_loss = max_loss
        self.random_seed = random_seed

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        learning_rate: float = 1e-4,
        epochs: int = 100,
    ):
        """
        Perform model learning for training dataset

        Args:
            X_train (np.ndarray): train samples decision matrix
            y_train (np.ndarray): train samples ground-truth value
            epochs (int): Number of train loops for fit method
            learning_rate (float): SGD learning rate

        """
        num_weights = X_train.shape[1] + 1
        np.random.seed(self.random_seed)
        self.weights = np.random.rand(num_weights) * 2 - 1
        for _ in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            for X_sample, y_sample in zip(X_train[permutation], y_train[permutation]):
                y_pred = self.predict(X_sample.reshape(1, -1))
                if self.l2_reg != 0:
                    l2_norm = np.linalg.norm(self.weights, ord=2)
                else:
                    l2_norm = 0
                if self.l1_reg != 0:
                    l1_norm = np.linalg.norm(self.weights, ord=1)
                else:
                    l1_norm = 0
                mse_der = 2 * (y_pred - y_sample)
                loss = np.clip(
                    mse_der + self.l2_reg * l2_norm + self.l1_reg * l1_norm,
                    -self.max_loss,
                    self.max_loss
                )
                grad_b = learning_rate * loss
                grad_w = learning_rate * loss * X_sample
                self.weights[0] = self.weights[0] - grad_b
                self.weights[1:] = self.weights[1:] - grad_w

    def predict(self, X: np.ndarray):
        """
        Perform model prediction for array of samples

        Args:
            X (np.ndarray): samples decision matrix

        """
        X_with_bias = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)
        y_pred = np.sum(self.weights * X_with_bias, axis=1)
        return y_pred
