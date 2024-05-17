# src/different_algorithms.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# Load the prepared data
df = pd.read_csv('data/prepared_customer_data.csv')

# Separate features and target variable
features = df.drop(columns=['CustomerID', 'Churn'])
target = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Initialize and train models
rf = RandomForestClassifier()
svc = SVC()
gb = GradientBoostingClassifier()

rf.fit(X_train, y_train)
svc.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_gb = gb.predict(X_test)

# Print performance metrics
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Support Vector Machine Accuracy:", accuracy_score(y_test, y_pred_svc))
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
