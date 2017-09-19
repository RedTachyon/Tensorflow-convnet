import numpy as np
from scipy import io

def normalize(data):
    """
    Normalizes the input data to 0 mean and 1 variance.
    
    Args:
        data: np.ndarray
    
    Returns:
        normalized data, mean, std
    """
    mean = data.mean()
    std = data.std()
    return (data - mean) / std, mean, std

def extract_and_norm_data(path):
    """
    Extracts and normalizes train data from the given path.
    
    Args:
        path: str, path to the data file
        
    Returns:
        X: np.ndarray, containing the normalized data
        mean: mean of the normalization
        std: std of the normalization
        labels: np.ndarray, containing the labels
    """
    all_data = io.loadmat(path)
    data, labels = all_data['X'], all_data['y']
    X, mean, std = normalize(data)
    X = X.transpose([3, 0, 1, 2])
    labels[labels == 10] = 0
    return X, mean, std, labels

def extract_test_data(path, mean, std):
    """
    Extracts and normalizes test data.
    
    Args:
        path: str, path to the data file
        mean: float, mean of the data for normalization
        std: float, std of the data for normalization
        
    Returns:
        X: np.ndarray, containing the normalized test data
        labels: np.ndarray, containing the test labels
    """
    all_data = io.loadmat(path)
    data, labels = all_data['X'], all_data['y']
    X = (data - mean) / std
    X = X.transpose([3, 0, 1, 2])
    labels[labels == 10] = 0
    return X, labels