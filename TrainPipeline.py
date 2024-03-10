import os
from src.DataIngestion import DataIngestion
from src.DataProcessing import DataProcessing
from src.ModelTraining import ClassifierTuner

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pickle


classifiers = {
    'Logistic Regression': (LogisticRegression(), {'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}),
    # 'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    # 'Random Forest': (RandomForestClassifier(), {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['sqrt', 'log2']}),
    # 'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
    # 'Gradient Boosting': (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
    # 'SVM': (SVC(), {'C': [ 0.1, 1, 10, 100], 'kernel': ['poly', 'rbf', 'sigmoid'], 'gamma': ['scale']}),
    # 'k-NN': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}),
    # 'Naive Bayes': (GaussianNB(), {}),
    # 'Neural Network': (MLPClassifier(), {'hidden_layer_sizes': [(100,), (50, 100), (50, 50, 50)], 'activation': ['relu', 'logistic'], 'solver': ['adam'], 'alpha': [0.0001, 0.001, 0.01], 'learning_rate': ['constant', 'invscaling', 'adaptive']}),
    # 'Linear Discriminant Analysis': (LinearDiscriminantAnalysis(), {}),
    # 'Quadratic Discriminant Analysis': (QuadraticDiscriminantAnalysis(), {}),
    # 'SGD Classifier': (SGDClassifier(), {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 'alpha': [0.0001, 0.001, 0.01, 0.1, 1]}),
    # 'XGBoost': (XGBClassifier(error_score='raise'), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
    # 'LightGBM': (LGBMClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
    # 'CatBoost': (CatBoostClassifier(), {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'depth': [3, 5, 7]})
}


if __name__ == "__main__":

    data = DataIngestion().load_data()
    x_train, x_test, y_train, y_test = DataProcessing(data).preprocess_data_for_target()
    classifier_tuner = ClassifierTuner(classifiers)
    classifier_tuner.tune_and_evaluate(x_train, y_train, x_test, y_test)
    classifier_tuner.save_models()

