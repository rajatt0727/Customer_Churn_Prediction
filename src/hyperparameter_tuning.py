# src/hyperparameter_tuning.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

# Load the prepared data
df = pd.read_csv('data/prepared_customer_data.csv')

# Separate features and target variable
features = df.drop(columns=['CustomerID', 'Churn'])
target = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Define parameter grid for Random Forest
param_grid_rf = {'n_estimators': [100, 200, 300],
                 'max_depth': [None, 5, 10, 15]}

# Perform random search for Random Forest
random_search_rf = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid_rf, n_iter=10, cv=5, random_state=42)
random_search_rf.fit(X_train, y_train)

# Get best parameters for Random Forest
best_params_rf = random_search_rf.best_params_

# Initialize and train Random Forest with best parameters
rf_best = RandomForestClassifier(**best_params_rf)
rf_best.fit(X_train, y_train)

# Make predictions
y_pred_rf_best = rf_best.predict(X_test)

# Print performance metrics for Random Forest with best parameters
print("Random Forest (Best Parameters) Accuracy:", accuracy_score(y_test, y_pred_rf_best))
