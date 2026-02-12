# data_loader.py

import pandas as pd
import numpy as np


def load_kaggle_data(filepath):
    """
    Load dataset from Kaggle.
    Args:
        filepath (str): The path to the Kaggle dataset file.
    Returns:
        pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    data = pd.read_csv(filepath)
    return data


def preprocess_data(df):
    """
    Preprocess the dataset (example: fill missing values, normalize data).
    Args:
        df (pd.DataFrame): The input dataframe.
    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Example preprocessing: fill missing values with the mean
    return df.fillna(df.mean())


def load_synthetic_data(sample_size=1000):
    """
    Generate synthetic dataset for testing.
    Args:
        sample_size (int): Number of samples to generate.
    Returns:
        pd.DataFrame: Generated synthetic dataset as a Pandas DataFrame.
    """
    synthetic_data = pd.DataFrame({
        'feature1': np.random.rand(sample_size),
        'feature2': np.random.rand(sample_size),
        'label': np.random.randint(0, 2, size=sample_size)
    })
    return synthetic_data
