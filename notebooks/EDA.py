import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore')
sns.set()
plt.style.use("ggplot")
#%matplotlib inline

# Load the dataset
data = pd.read_csv('CDC-Diabetes-Indicators-Analysis/data/diabetes.csv')
# Display basic information about the dataset
#check the data
print(data.head())
print(data.info())
print(data.describe())
print(data.columns)
# Visualize distributions
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.show()


# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
# distribution of outcome variable
print(data.Outcome.value_counts()*100/len(df))

# plot the hist of the age variable
plt.figure(figsize=(8,7))
plt.xlabel('Age', fontsize=10)
plt.ylabel('Count', fontsize=10)
print(data['Age'].hist(edgecolor="black"))
print(data['age'].max())
print(data['Age'].min())

# density graph
# 4*2=8
# columns=2 figure
# having 4 row

# [0,0], [0,1]
# [1,0], [1,1]
# [2,0], [2,1]
# [3,0], [3,1]

fig,ax = plt.subplots(4,2, figsize=(20,20))
sns.distplot(data.Pregnancies, bins=20, ax=ax[0,0], color="red")
sns.distplot(data.Glucose, bins=20, ax=ax[0,1], color="red")
sns.distplot(data.BloodPressure, bins=20, ax=ax[1,0], color="red")
sns.distplot(data.SkinThickness, bins=20, ax=ax[1,1], color="red")
sns.distplot(data.Insulin, bins=20, ax=ax[2,0], color="red")
sns.distplot(data.BMI, bins=20, ax=ax[2,1], color="red")
sns.distplot(data.DiabetesPedigreeFunction, bins=20, ax=ax[3,0], color="red")
sns.distplot(data.Age, bins=20, ax=ax[3,1], color="red")

data.groupby("Outcome").agg({'Pregnancies':'mean'})
data.groupby("Outcome").agg({'Pregnancies':'max'})
data.groupby("Outcome").agg({'Glucose':'mean'})
data.groupby("Outcome").agg({'Glucose':'max'})

# Additional EDA
# 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age'
#     groupby-> mean/max

f,ax = plt.subplots(1,2, figsize=(18,8))
data['Outcome'].value_counts().plot.pie(explode=[0,0.1],autopct = "%1.1f%%", ax=ax[0], shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('Outcome', data=data, ax=ax[1])
ax[1].set_title('Outcome')
print(plt.show())

# pair plot
p = sns.pairplot(data, hue="Outcome")

# Outlier Detection
# IQR+Q1
# 50%
# 24.65->25%+50%
# 24.65->25%
for feature in data:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3-Q1
    lower = Q1-1.5*IQR
    upper = Q3+1.5*IQR
    if df[(df[feature]>upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")
        

plt.figure(figsize=(8,7))
sns.boxplot(x= data["Insulin"], color="red")