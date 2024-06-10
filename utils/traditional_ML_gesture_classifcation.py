from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def knn_accuracy_vs_num_neighbors(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    accuracies = []

    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        accuracies.append(accuracy)
        print(f'k={k}, Accuracy: {accuracy*100:.2f}%')

    # Plot the accuracies for each k
    plt.plot(k_range, accuracies, marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.title('KNN Accuracy vs Number of Neighbors')
    plt.grid(True)
    plt.show()
    
    return accuracies


def svm_hyperparameter_tuning(X_train, y_train, X_test, y_test, kernels=['linear', 'rbf', 'poly', 'sigmoid'], use_condensed_grid=True):
    """
    Train SVM classifiers with different hyperparameters and plot the accuracy for each kernel.

    Parameters:
    - X_train: Training data embeddings
    - y_train: Training labels
    - X_test: Test data embeddings
    - y_test: Test labels
    - kernels: List of kernel types to test (default: ['linear', 'rbf', 'poly', 'sigmoid'])

    Returns:
    - best_params: Best parameters found for each kernel type
    - best_accuracies: Best accuracy found for each kernel type
    """

    print("Starting")

    if use_condensed_grid:
        # CONDENSED FOR SPEED...
        C_lst = [0.1, 1, 10]
        gamma_lst = [1, 0.1]
        param_grid = {
            'linear': {'C': C_lst},
            'rbf': {'C': C_lst, 'gamma': gamma_lst},
            'poly': {'C': C_lst, 'degree': [2, 3], 'gamma': gamma_lst, 'coef0': [0, 1]},
            'sigmoid': {'C': C_lst, 'gamma': gamma_lst, 'coef0': [0, 1]}
        }
    else:
        # FULL
        C_lst = [0.1, 1, 10, 100]
        gamma_lst = [1, 0.1, 0.01, 0.001]
        param_grid = {
            'linear': {'C': C_lst},
            'rbf': {'C': C_lst, 'gamma': gamma_lst},
            'poly': {'C': C_lst, 'degree': [2, 3, 4], 'gamma': gamma_lst, 'coef0': [0, 1, 10]},
            'sigmoid': {'C': C_lst, 'gamma': gamma_lst, 'coef0': [0, 1, 10]}
        }
        
    best_params = {}
    best_accuracies = {}
    
    for kernel in kernels:
        grid = GridSearchCV(SVC(kernel=kernel, random_state=42), param_grid[kernel], refit=True, verbose=0)
        grid.fit(X_train, y_train)
        
        y_pred = grid.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        best_params[kernel] = grid.best_params_
        best_accuracies[kernel] = accuracy
        print(f'Kernel={kernel}, Best Params: {grid.best_params_}, Accuracy: {accuracy*100:.2f}%')
        #print(f"Classification Report for {kernel} kernel:\n", classification_report(y_test, y_pred))
    
    plt.bar(best_accuracies.keys(), best_accuracies.values())
    plt.xlabel('Kernel Type')
    plt.ylabel('Accuracy')
    plt.title('SVM Accuracy vs Kernel Type')
    plt.show()
    
    return best_params, best_accuracies