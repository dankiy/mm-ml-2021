from abc import abstractmethod, ABC


class Model(ABC):
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Perform model learning for training dataset

        Args:
            X_train (np.ndarray): train samples decision matrix
            y_train (np.ndarray): train samples ground-truth value

        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Perform model prediction for array of samples

        Args:
            X (np.ndarray): samples decision matrix

        """
        pass
