import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the preprocessed data
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv').values.ravel()

# Initialize and train the model (example: Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, '../models/logistic_regression_model.pkl')

print("Model training completed and saved to 'logistic_regression_model.pkl'.")
