### Logistic Regression
import numpy as np
import pandas as pd
import category_encoders as ce

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from hmeasure import h_score


# Importing datasets from dataset.py (see this file for more info)
from dataset import german_dataset, credit_risk_dataset, hmeq_dataset, default_dataset
### Choose one of the datasets
dataset = german_dataset
# Read dataset and drop NAs if necessary
df = pd.read_csv(dataset.filename, delimiter=';')
df.dropna()
# Defining X and y columns
train_columns = [col for col in df.columns if col != dataset.goal_column]
X, y = df[train_columns], df[dataset.goal_column]
# Numerical and Categorical variables treatment
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('woe', ce.WOEEncoder())])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, dataset.category_columns),
    ('num', numeric_transformer, dataset.numeric_columns)])

pipe = Pipeline(steps=[('preprocessor', preprocessor)])

target = y.to_numpy()
X_trans = pipe.fit_transform(X, target)
df_trans = pd.DataFrame(X_trans, columns=dataset.get_vars())

# Train-test split of 80%-20%
X_train, X_test, y_train, y_test = train_test_split(df_trans, target, test_size=0.2, random_state=123)

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

# Values of the parameters of LR
print(logreg.coef_)
# Confusion matrix of LR
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
# AUC and ROC curve
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
print("The AUC value of Logistic Regression is: ")
print(auc)
# H-measure
H = h_score(y_test, y_pred)
print("The H-measure is: ")
print(H)
# EMPCS value of LR is obtained through EMP package in R


