import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('CDC-Diabetes-Indicators-Analysis/data/diabetes.csv')

# Handle missing values (example: fill with median)
data.fillna(data.median(), inplace=True)

# Encode categorical variables if any (example code, adjust as necessary)
# data['CategoricalColumn'] = data['CategoricalColumn'].astype('category').cat.codes

# Split the data into features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data to CSV files
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

print("Data preprocessing completed and saved to CSV files.")
