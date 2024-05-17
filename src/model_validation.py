# src/model_validation.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load the prepared data
df = pd.read_csv('data/prepared_customer_data.csv')

# Separate features and target
features = df.drop(columns=['CustomerID', 'Churn'])
target = df['Churn']

# Scale the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Logistic Regression with increased max_iter and scaled data
logreg = LogisticRegression(max_iter=1000)
logreg_cv_scores = cross_val_score(logreg, scaled_features, target, cv=5)
print("Logistic Regression CV Scores:", logreg_cv_scores)

# Decision Tree with tuned hyperparameters and scaled data
tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, min_samples_leaf=2)
tree_cv_scores = cross_val_score(tree, scaled_features, target, cv=5)
print("Decision Tree CV Scores:", tree_cv_scores)
