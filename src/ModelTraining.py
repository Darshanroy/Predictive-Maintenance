import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import mlflow


class ClassifierTuner:
    def __init__(self, classifiers):
        self.classifiers = classifiers
        self.best_models = {}

    def tune_hyperparameters(self, name, classifier, param_grid, X_train, y_train, X_test, y_test):
        # Ignore warnings
        warnings.filterwarnings('ignore')

        # Create GridSearchCV object
        grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        # Fit the GridSearchCV object to find the best parameters
        grid_search.fit(X_train, y_train)

        # Reset warning behavior
        warnings.resetwarnings()

        # Print the best parameters found
        print("Best Parameters for", name, ":", grid_search.best_params_)

        # Print the best score achieved during hyperparameter tuning
        print("Best Score for", name, ":", grid_search.best_score_)

        # Evaluate the performance of the best model on the test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy for", name, ":", test_accuracy)

        # Save the best model
        self.best_models[name] = best_model

        # Log parameters and metrics to MLflow
        with mlflow.start_run() as run:
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_score", grid_search.best_score_)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.sklearn.log_model(best_model, f"{name}_best_model")

    def save_models(self):
        for name, model in self.best_models.items():
            model_filename = f"trained_models/{name}_best_model.pkl"
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
            print(f"Best parameterized model for {name} dumped to {model_filename}")

    def tune_and_evaluate(self, X_train, y_train, X_test, y_test):

        mlflow.set_tracking_uri("http://localhost:5000")  # For a local MLflow server
        mlflow.set_experiment("Predictive Maintenance")

        for name, (classifier, param_grid) in self.classifiers.items():
            print("Tuning hyperparameters for:", name)
            self.tune_hyperparameters(name, classifier, param_grid, X_train, y_train, X_test, y_test)
