import numpy as np

from ..abstract import Model


class AffProp(Model):
    def __init__(
        self, noise: float = 1e-16, discount: float = 0.5, random_seed: int = 0
    ):
        """
        Affinity propagation clustering

        Args:
            noise (float): size of noise for similarity matrix
            discount (float): averaging discount value
            random_seed (int): value for numpy random number generator initialization

        Methods:
            fit (X_train, y_train, learning_rate=1e-4, epochs=100): perform clustering
                        for input data
            predict (X): not implemented

        """
        self.noise = noise
        self.discount = discount
        self.random_seed = 0

    def fit(self, X: np.ndarray, num_iters: int = 50):
        """
        Perform clustering for input dataset, result is stored in labels_

        Args:
            X (np.ndarray): input similarity matrix
            num_iters (int): number of algorithm iterations

        """
        np.random.seed(self.random_seed)
        S = X.copy().astype(float)
        num_samples = S.shape[0]
        R = np.zeros((num_samples, num_samples))
        A = np.zeros((num_samples, num_samples))
        temp = np.zeros((num_samples, num_samples))
        for row_idx in range(num_samples):
            S[row_idx, :] += np.random.rand(num_samples) * self.noise

        ids = np.arange(num_samples)

        for _ in range(num_iters):
            temp = A + S
            I1 = np.argmax(temp, axis=1)
            Y1 = temp[ids, I1]

            temp[ids, I1] = -np.inf
            Y2 = np.max(temp, axis=1)

            temp = S - Y1[:, None]
            temp[ids, I1] = S[ids, I1] - Y2
            temp *= 1 - self.discount
            R *= self.discount
            R += temp

            temp = np.maximum(R, 0)
            temp.flat[:: num_samples + 1] = R.flat[:: num_samples + 1]
            temp -= np.sum(temp, axis=0)
            dA = np.diag(temp).copy()

            temp = np.clip(temp, 0, np.inf)
            temp.flat[:: num_samples + 1] = dA
            temp *= 1 - self.discount
            A *= self.discount
            A -= temp
            E = (np.diag(A) + np.diag(R)) > 0
            K = np.sum(E, axis=0)

        I2 = np.flatnonzero(E)
        K = I2.size
        c = np.argmax(S[:, I2], axis=1)
        c[I2] = np.arange(K)

        for k in range(K):
            diag = np.where(c == k)[0]
            j = np.argmax(np.sum(S[diag[:, np.newaxis], diag], axis=0))
            I2[k] = diag[j]

        c = np.argmax(S[:, I2], axis=1)
        c[I2] = np.arange(K)
        interim_labels = I2[c]

        self.cluster_centers_indices_ = np.unique(interim_labels)
        self.labels_ = np.searchsorted(self.cluster_centers_indices_, interim_labels)

    def predict(self, X):
        """
        Not implemented
        """
        pass
