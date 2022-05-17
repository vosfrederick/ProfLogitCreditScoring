### Decision tree
import numpy as np
import pandas as pd
import category_encoders as ce

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from hmeasure import h_score
import matplotlib.pyplot as plt
# Importing datasets from dataset.py (see this file for more info)
from dataset import german_dataset, credit_risk_dataset, hmeq_dataset, default_dataset
# Choose a dataset
dataset = german_dataset
# Read data and drop NAs
df = pd.read_csv(dataset.filename, delimiter=';')
df.dropna()
# Define X and y columns
train_columns = [col for col in df.columns if col != dataset.goal_column]
X, y = df[train_columns], df[dataset.goal_column]
# Treatment of numerical and categorical variables
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[('woe', ce.WOEEncoder())])
preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, dataset.category_columns),
    ('num', numeric_transformer, dataset.numeric_columns)
])
pipe = Pipeline(steps=[('preprocessor', preprocessor)])
target = y.to_numpy()
X_trans = pipe.fit_transform(X, target)
df_trans = pd.DataFrame(X_trans, columns=dataset.get_vars())
#Splitting training and test data 80%-20%
X_train, X_test, y_train, y_test = train_test_split(df_trans, target, test_size=0.2, random_state=123)
print(X_train)

# Decision tree algorithm with parameters based on grid search below (Vla
credit_tree = DecisionTreeClassifier(criterion='entropy',max_depth = 8, min_samples_leaf = 5, min_samples_split = 2)
credit_tree.fit(X_train, y_train)
predictions = credit_tree.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, predictions)
print(cnf_matrix)

### Grid Search to determine optimal parameter value for each tree
param= {"max_depth": [2,4,6,8,10],
        "min_samples_split":[2,4,6,8,10],
        "min_samples_leaf":[1,3,5]}

grid = GridSearchCV(credit_tree,param_grid=param)
grid.fit(X_train,y_train)
print(grid.best_params_)

# AUC and ROC for DT
y_pred_probaa = credit_tree.predict_proba(X_test)[::,1]
fprr, tprr, _ = metrics.roc_curve(y_test,  y_pred_probaa)
auc1 = metrics.roc_auc_score(y_test, y_pred_probaa)
plt.plot(fprr,tprr,label="data 1, auc="+str(auc1))
plt.legend(loc=4)
plt.show()
print("The AUC value of Decision Tree is: ")
print(auc1)
# H-measure for DT
H = h_score(y_test, predictions)
print("The H-measure is: ")
print(H)
# EMPCS value of DT is obtained through EMP package on R