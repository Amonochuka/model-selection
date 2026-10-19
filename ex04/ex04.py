#!/usr/bin/env python3
"""
Exercise 4: Validation curve and Learning curve
Plot validation curve for max_depth and learning curve for RandomForestClassifier.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, learning_curve
import time

def plot_validation_curve():
    """Plot validation curve for max_depth parameter."""
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=100000,
        n_features=30,
        n_informative=10,
        flip_y=0.2,
        random_state=42
    )
    
    print("Computing validation curve...")
    param_range = np.arange(1, 21)
    
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42, n_estimators=50),
        X, y,
        param_name="max_depth",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label="Training score", color="darkorange", lw=2)
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                     alpha=0.2, color="darkorange")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="navy", lw=2)
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                     alpha=0.2, color="navy")
    
    plt.title("Validation Curve for RandomForestClassifier (max_depth)")
    plt.xlabel("max_depth")
    plt.ylabel("Score (Accuracy)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.savefig("ex04/w2_day5_ex5_q1.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved validation curve to ex04/w2_day5_ex5_q1.png")

def plot_learning_curve():
    """Plot learning curve for RandomForestClassifier with max_depth=12."""
    print("Generating dataset...")
    X, y = make_classification(
        n_samples=100000,
        n_features=30,
        n_informative=10,
        flip_y=0.2,
        random_state=42
    )
    
    print("Computing learning curve...")
    clf = RandomForestClassifier(max_depth=12, random_state=42, n_estimators=50)
    
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        clf, X, y,
        cv=10,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        return_times=True
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Learning curve
    axes[0].plot(train_sizes, train_mean, label="Training score", color="darkorange", lw=2)
    axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.2, color="darkorange")
    axes[0].plot(train_sizes, test_mean, label="Cross-validation score", color="navy", lw=2)
    axes[0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                         alpha=0.2, color="navy")
    axes[0].set_title("Learning Curve")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score (Accuracy)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scalability
    axes[1].plot(train_sizes, fit_times_mean, label="Fit time", color="green", lw=2)
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.2, color="green")
    axes[1].set_title("Scalability of the model")
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Performance
    axes[2].plot(fit_times_mean, test_mean, label="Performance", color="red", lw=2)
    axes[2].fill_between(fit_times_mean, test_mean - test_std, test_mean + test_std,
                         alpha=0.2, color="red")
    axes[2].set_title("Performance of the model")
    axes[2].set_xlabel("Fit time (s)")
    axes[2].set_ylabel("Score (Accuracy)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ex04/w2_day5_ex5_q2.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved learning curve to ex04/w2_day5_ex5_q2.png")

def main():
    plot_validation_curve()
    plot_learning_curve()
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()