import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('../data/diabetes.csv')

# Display basic information about the dataset
print(data.info())
print(data.describe())

# Visualize distributions
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

sns.countplot(x='Outcome', data=data)
plt.title('Outcome Count')
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
