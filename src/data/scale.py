import numpy as np


def minmax(
    train: np.ndarray,
    test: np.ndarray,
    min_value: float = 0,
    max_value: float = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale train and test feature arrays inpedendently for each feature
    by min and max values

    Args:
        train (np.ndarray): train feature array
        test (np.ndarray): test feature array
        min_value (float): chosen minimal feature value
        max_value (float): chosen maximal feature value
    Returns:
        tuple[np.ndarray, np.ndarray]: scaled train and test feature arrays

    """
    eps = 1e-12
    train_min = np.min(train, axis=0)
    train_max = np.max(train, axis=0)

    train_standardized = (train - train_min) / np.clip(train_max - train_min, eps, None)
    train_scaled = train_standardized * (max_value - min_value) + min_value

    test_standardized = (test - train_min) / np.clip(train_max - train_min, eps, None)
    test_scaled = test_standardized * (max_value - min_value) + min_value

    return train_scaled, test_scaled


def standard(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardize train and test feature arrays independently for each feature
    by mean and std

    Args:
        train (np.ndarray): train feature array
        test (np.ndarray): test feature array
        mean_value (float): chosen minimal feature value
        std_value (float): chosen maximal feature value
    Returns:
        tuple[np.ndarray, np.ndarray]: scaled train and test feature arrays

    """
    eps = 1e-12
    train_mean = np.mean(train, axis=0)
    train_std = np.std(train, axis=0)

    train_standardized = (train - train_mean) / np.clip(train_std, eps, None)
    test_standardized = (test - train_mean) / np.clip(train_std, eps, None)

    return train_standardized, test_standardized
