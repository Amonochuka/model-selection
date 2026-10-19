#!/usr/bin/env python3
"""
Exercise 1: K-Fold Cross-Validation
Using KFold to split the dataset into a k-fold cross-validation.
"""

import numpy as np
from sklearn.model_selection import KFold

def main():
    # Given data
    X = np.array(np.arange(1, 21).reshape(10, -1))
    y = np.array(np.arange(1, 11))
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("X:\n", X)
    print("y:\n", y)
    
    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=False)
    
    fold_num = 1
    for train_index, test_index in kf.split(X):
        print(f"\nFold:  {fold_num}")
        print(f"TRAIN: {train_index} TEST: {test_index}")
        fold_num += 1

if __name__ == "__main__":
    main()