# src/eda.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/prepared_customer_data.csv')

# Exclude non-numeric columns from correlation calculation
numeric_cols = df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols]

# Distribution of Churn
sns.countplot(x='Churn', data=df)
plt.show()

# Correlation Matrix
corr_matrix = df_numeric.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
