# import pickle
# from sklearn.metrics import accuracy_score
#
#
# class ModelEvaluator:
#     def __init__(self, best_models, X_test, y_test):
#         self.best_models = best_models
#         self.X_test = X_test
#         self.y_test = y_test
#
#     def evaluate_models(self):
#         evaluation_results = {}
#         for name, model in self.best_models.items():
#             y_pred = model.predict(self.X_test)
#             test_accuracy = accuracy_score(self.y_test, y_pred)
#             print(f"Test Accuracy for {name}: {test_accuracy}")
#             evaluation_results[name] = test_accuracy
#         return evaluation_results
#
#     def save_models(self):
#         for name, model in self.best_models.items():
#             model_filename = f"{name}_best_model.pkl"
#             with open(model_filename, 'wb') as file:
#                 pickle.dump(model, file)
#             print(f"Best parameterized model for {name} dumped to {model_filename}")
