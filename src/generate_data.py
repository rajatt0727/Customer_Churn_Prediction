# src/generate_data.py
import pandas as pd
from faker import Faker
import random

fake = Faker()
Faker.seed(0)
random.seed(0)

num_customers = 1000
data = []

for _ in range(num_customers):
    customer_id = fake.uuid4()
    subscription_length = random.randint(1, 60)  # months
    last_login = fake.date_between(start_date='-2y', end_date='today')
    support_tickets = random.randint(0, 10)
    payment_history = random.choice(['on-time', 'late', 'delayed'])
    age = random.randint(18, 70)
    gender = random.choice(['Male', 'Female'])
    churn = random.choice([0, 1])  # 0 for active, 1 for churned

    data.append([customer_id, subscription_length, last_login, support_tickets, payment_history, age, gender, churn])

columns = ['CustomerID', 'SubscriptionLength', 'LastLogin', 'SupportTickets', 'PaymentHistory', 'Age', 'Gender', 'Churn']
df = pd.DataFrame(data, columns=columns)

df.to_csv('data/synthetic_customer_data.csv', index=False)
