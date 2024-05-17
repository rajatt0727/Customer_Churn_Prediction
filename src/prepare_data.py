# src/prepare_data.py
import pandas as pd

df = pd.read_csv('data/synthetic_customer_data.csv')

df['PaymentHistory'] = df['PaymentHistory'].map({'on-time': 0, 'late': 1, 'delayed': 2})
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['LastLogin'] = pd.to_datetime(df['LastLogin'])
df['DaysSinceLastLogin'] = (pd.to_datetime('today') - df['LastLogin']).dt.days
df.drop(columns=['LastLogin'], inplace=True)

df.to_csv('data/prepared_customer_data.csv', index=False)
