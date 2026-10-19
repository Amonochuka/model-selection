#!/usr/bin/env python3
"""
Exercise 3: GridSearchCV
Run GridSearchCV with RandomForestRegressor, optimize max_depth and n_estimators.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def main():
    # Load data
    housing = fetch_california_housing()
    X, y = housing['data'], housing['target']
    
    # Split data train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=43
    )
    
    # Define parameter grid
    param_grid = {
        'max_depth': [5, 10, 15, 20],
        'n_estimators': [10, 50, 100]
    }
    
    # Create GridSearchCV
    rf = RandomForestRegressor(random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    print("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    # Best estimator
    best_estimator = grid_search.best_estimator_
    print("\nBest parameters:", grid_search.best_params_)
    print("Best validation score (neg MSE):", grid_search.best_score_)
    print("Best validation MSE:", -grid_search.best_score_)
    
    # CV results
    print("\nCV Results:")
    print(grid_search.cv_results_)
    
    # Test set score
    test_score = best_estimator.score(X_test, y_test)
    print(f"\nTest set R2 score: {test_score}")
    
    # Test set MSE
    from sklearn.metrics import mean_squared_error
    y_pred = best_estimator.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    print(f"Test set MSE: {test_mse}")

if __name__ == "__main__":
    main()