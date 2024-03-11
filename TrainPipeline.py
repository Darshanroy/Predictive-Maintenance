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
import os

# Parameters are Fine-tuned and set a correct parameters to Decrease computation
classifiers = {
    # 'Logistic Regression': (LogisticRegression(),  {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}),
    # 'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    # 'Random Forest': (RandomForestClassifier(), {'criterion': ['entropy'], 'max_depth':[15], 'min_samples_leaf':[ 2], 'min_samples_split': [5]}),
    # 'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [100], 'learning_rate': [1.0]}),
    # 'Gradient Boosting': (GradientBoostingClassifier(), {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}),
    # 'SVM': (SVC(), {'C': 100, 'gamma': 'scale', 'kernel': 'poly'}),
    # 'k-NN': (KNeighborsClassifier(), {'algorithm': 'auto', 'n_neighbors': 3, 'weights': 'distance'}{'algorithm': 'auto', 'n_neighbors': 3, 'weights': 'distance'}),
    # 'Naive Bayes': (GaussianNB(), {}),
    # 'Neural Network': (MLPClassifier(), {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (100,), 'learning_rate': 'adaptive', 'solver': 'adam'}),
    # 'Linear Discriminant Analysis': (LinearDiscriminantAnalysis(), {}),
    # 'Quadratic Discriminant Analysis': (QuadraticDiscriminantAnalysis(), {}),
    'SGD Classifier': (SGDClassifier(),  {'alpha': [1], 'loss': ['squared_hinge'], 'penalty': ['elasticnet']}),
    'XGBoost': (XGBClassifier(error_score='raise'), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
    'LightGBM': (LGBMClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 5, 7]}),
    'CatBoost': (CatBoostClassifier(),  {'depth': [5], 'iterations': [100], 'learning_rate': [1.0]})
}


if __name__ == "__main__":

    data = DataIngestion().load_data()
    x_train, x_test, y_train, y_test = DataProcessing(data).preprocess_data_for_target()
    classifier_tuner = ClassifierTuner(classifiers)
    classifier_tuner.tune_and_evaluate(x_train, y_train, x_test, y_test)
    classifier_tuner.save_models()

