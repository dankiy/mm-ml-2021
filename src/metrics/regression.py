import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error

    Args:
        y_true (np.ndarray): ground-truth values
        y_pred (np.ndarray): y_predicted values
    Returns:
        float: metric value

    """
    result = np.mean((y_true - y_pred) ** 2)

    return result


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error

    Args:
        y_true (np.ndarray): ground-truth values
        y_pred (np.ndarray): y_predicted values
    Returns:
        float: metric value

    """
    result = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return result


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination

    Args:
        y_true (np.ndarray): ground-truth values
        y_pred (np.ndarray): y_predicted values
    Returns:
        float: metric value

    """
    result = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2) # noqa

    return result
