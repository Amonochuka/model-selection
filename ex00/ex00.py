#!/usr/bin/env python3
"""
Exercise 0: Environment and libraries
Verify that all required libraries are installed and working.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
import sklearn

def check_versions():
    print("Python version:", sys.version)
    print("NumPy version:", np.__version__)
    print("Pandas version:", pd.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Scikit-learn version:", sklearn.__version__)
    
    # Quick functionality test
    arr = np.array([1, 2, 3, 4, 5])
    print("\nNumPy test - array sum:", arr.sum())
    
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    print("Pandas test - DataFrame shape:", df.shape)
    
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    print("Scikit-learn test - imports successful")

if __name__ == "__main__":
    check_versions()
    print("\n✓ All libraries installed and working correctly!")