#!/usr/bin/env python3
"""
Exercise 2: Cross-validation (k-fold)
Cross validate a Linear Regression pipeline using cross_validate with 10 folds.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import numpy as np

def main():
    # Load data
    housing = fetch_california_housing()
    X, y = housing['data'], housing['target']
    
    # Split data train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=43
    )
    
    # Create pipeline
    pipeline = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ]
    pipe = Pipeline(pipeline)
    
    # Cross validate with 10 folds
    cv_results = cross_validate(pipe, X_train, y_train, cv=10, scoring='r2')
    
    scores = cv_results['test_score']
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    print("Scores on validation sets:")
    print(scores)
    print()
    print("Mean of scores on validation sets:")
    print(f" {mean_score}")
    print()
    print("Standard deviation of scores on validation sets:")
    print(f" {std_score}")

if __name__ == "__main__":
    main()