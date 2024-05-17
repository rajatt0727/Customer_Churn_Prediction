import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load datasets
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').values.ravel()
y_test = pd.read_csv('data/y_test.csv').values.ravel()

# Logistic Regression with Hyperparameter Tuning
log_reg = LogisticRegression()
param_grid_log_reg = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search_log_reg = GridSearchCV(log_reg, param_grid_log_reg, cv=5, scoring='accuracy')
grid_search_log_reg.fit(X_train, y_train)

# Evaluate Logistic Regression
best_log_reg = grid_search_log_reg.best_estimator_
y_pred_log_reg = best_log_reg.predict(X_test)

print("Logistic Regression Model Performance")
print("-----------------------------")
print(f"Best Parameters: {grid_search_log_reg.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_log_reg))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))

# Save Logistic Regression Model
joblib.dump(best_log_reg, 'logistic_regression_model.pkl')

# Decision Tree with Hyperparameter Tuning
tree_clf = DecisionTreeClassifier()
param_grid_tree = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
grid_search_tree = GridSearchCV(tree_clf, param_grid_tree, cv=5, scoring='accuracy')
grid_search_tree.fit(X_train, y_train)

# Evaluate Decision Tree
best_tree = grid_search_tree.best_estimator_
y_pred_tree = best_tree.predict(X_test)

print("\nDecision Tree Model Performance")
print("-----------------------------")
print(f"Best Parameters: {grid_search_tree.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_tree))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

# Save Decision Tree Model
joblib.dump(best_tree, 'decision_tree_model.pkl')

print("\nModels have been trained, evaluated, and saved successfully.")
