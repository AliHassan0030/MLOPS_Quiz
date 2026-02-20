import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Ensure directory exists [cite: 33]
os.makedirs('data/processed', exist_ok=True)

# 1. Load dataset [cite: 30]
# Use the correct path relative to your script
data = pd.read_csv('titanic.csv')

# 2. Handle missing values
# We only calculate the mean for numeric columns to avoid the TypeError
numeric_cols = data.select_dtypes(include=['number']).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# For string columns, fill with "Unknown" or the most frequent value
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

# 3. Split into train/test [cite: 32]
# Using 'Survived' as the target for the Titanic dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)

# 4. Save processed data [cite: 33]
train.to_csv('data/processed/train.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)
print("Data preprocessing complete. Files saved in data/processed/")