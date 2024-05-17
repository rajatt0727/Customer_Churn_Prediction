# src/feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

# Load the prepared data
df = pd.read_csv('data/prepared_customer_data.csv')

# Separate features and target variable
features = df.drop(columns=['CustomerID', 'Churn'])
target = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Feature selection with RFE and cross-validation
logreg = LogisticRegression(max_iter=1000)
rfe_logreg = RFECV(estimator=logreg, cv=5)
rfe_logreg.fit(X_train, y_train)
X_train_selected_logreg = rfe_logreg.transform(X_train)
X_test_selected_logreg = rfe_logreg.transform(X_test)

# Generate polynomial features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_selected_logreg)
X_test_poly = poly.transform(X_test_selected_logreg)

# Save the transformed data
pd.DataFrame(X_train_poly).to_csv('data/X_train_poly.csv', index=False)
pd.DataFrame(X_test_poly).to_csv('data/X_test_poly.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Confirm X_train_selected_logreg is defined correctly
print("X_train_selected_logreg is defined successfully!")
