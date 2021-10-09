import numpy as np


def train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split input data randomly after shuffling

    Args:
        X (np.ndarray): decision matrix
        y (np.ndarray): ground-truth labels
        test_size (float): fraction of test split
        random_seed (int): number to initialize a pseudorandom number generator

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, y_train,
                    X_test, y_test

    """
    np.random.seed(random_seed)
    num_samples = X.shape[0]
    num_train_samples = int(num_samples * (1 - test_size))
    permuted_ids = np.random.permutation(np.arange(num_samples))
    train_ids = permuted_ids[:num_train_samples]
    test_ids = permuted_ids[num_train_samples:]
    X_test = X[test_ids]
    X_train = X[train_ids]
    y_test = y[test_ids]
    y_train = y[train_ids]

    return X_train, y_train, X_test, y_test


class _KFoldIterator:
    def __init__(self, kfold):
        self._kfold = kfold
        self._counter = 0

    def __next__(self):
        if self._counter < self._kfold.num_folds:
            item = self._kfold.__getitem__(self._counter)
            self._counter += 1
            return item
        else:
            raise StopIteration


class KFold:
    """Iterable cross-validation object

    Args:
        X (np.ndarray): samples decision matrix
        y (np.ndarray): samples ground-truth value
        num_folds (int): number of cross-validation folds
        random_seed (int): value for numpy random number generator initialization

    Methods:
        __getitem__(key): returns X_train, y_train, X_test, y_test

    """
    def __init__(self, X: np.ndarray, y: np.ndarray, num_folds: int, random_seed: int):
        self.num_samples = X.shape[0]
        self.num_folds = num_folds
        np.random.seed(random_seed)
        permuted_ids = np.random.permutation(np.arange(self.num_samples))
        self.X = X[permuted_ids]
        self.y = y[permuted_ids]

    def __getitem__(self, key: int):
        assert key < self.num_folds, "Key must be lower than number of folds"
        assert key >= 0, "Key must be not negative"
        test_start_id = int(key * self.num_samples / self.num_folds)
        test_end_id = int((key + 1) * self.num_samples / self.num_folds)
        X_test = self.X[test_start_id: test_end_id]
        X_train = np.concatenate([
                self.X[: test_start_id],
                self.X[test_end_id:],
            ],
            axis=0,
        )
        y_test = self.y[test_start_id: test_end_id]
        y_train = np.concatenate([
                self.y[: test_start_id],
                self.y[test_end_id:],
            ],
            axis=0,
        )

        return X_train, y_train, X_test, y_test

    def __iter__(self):
        return _KFoldIterator(self)


def cross_val(
    X: np.ndarray,
    y: np.ndarray,
    num_folds: int,
    random_seed: int = 0,
) -> KFold:
    """
    Make cross-validation split randomly after shuffling

    Args:
        X (np.ndarray): decision matrix
        y (np.ndarray): ground-truth labels
        num_folds (int): number of train/test folds
        random_seed (int): number to initialize a pseudorandom number generator

    Returns:
        KFold: object containing data with __getitem__ method for getting splits

    """
    kfold = KFold(X, y, num_folds, random_seed)

    return kfold
