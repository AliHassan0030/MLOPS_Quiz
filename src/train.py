import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Load processed data
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')

# 2. Basic Feature Selection for Titanic
# We use numeric columns and 'Sex' (mapped to 0 and 1)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

X_train = train[features]
y_train = train[target]
X_test = test[features]
y_test = test[target]

# 3. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Save model in /models/ [cite: 40]
os.makedirs('models', exist_ok=True)
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 5. Print accuracy [cite: 41]
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Training Complete.")
print(f"Final Accuracy Score: {accuracy:.4f}")