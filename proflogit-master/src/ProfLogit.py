# Standard
import numpy as np
import pandas as pd
import category_encoders as ce

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn import metrics
# others
from hmeasure import h_score
import matplotlib.pyplot as plt
from proflogit.base import ProfLogitCS
from proflogit.utils import load_data
# Importing datasets from dataset.py (see this file for more info)
from dataset import german_dataset, credit_risk_dataset, hmeq_dataset , default_dataset
# Choose a dataset
dataset = german_dataset

DAT_DIR = '../data'
# Read dataset and drop NAs if needed
df = pd.read_csv(dataset.filename, delimiter=';')
df.dropna()
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[('woe', ce.WOEEncoder())])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, dataset.category_columns),
    ('num', numeric_transformer, dataset.numeric_columns)])

pipe = Pipeline(steps=[('preprocessor', preprocessor)])

#Splitting df into df_train and df_test with 80% and 20%
df_train, df_test = train_test_split(df, test_size=0.2, random_state=123)

#Define X_train and y_train and preprocess it
X_train, y_train = df_train.iloc[:,:-1], np.array(df_train.iloc[:,-1]).reshape(-1,1)

#fit_transform for Training data
X_transtrain = pipe.fit_transform(X = X_train,y=y_train)

df_transtrain = pd.DataFrame(X_transtrain, columns = dataset.get_vars())
df_transtrain[dataset.goal_column]=y_train

#Define X_test and y_test and preprocess it
X_test, y_test = df_test.iloc[:,:-1], np.array(df_test.iloc[:,-1]).reshape(-1,1)

##transform for test data
X_transtest = pipe.transform(X = X_test)

df_transtest = pd.DataFrame(X_transtest, columns = dataset.get_vars())
df_transtest[dataset.goal_column]= y_test

#Creating train and test dat files (these files will adapt to the dataset we use)
df_transtrain.to_csv(f'{DAT_DIR}/train_data.dat', sep=' ', index=False)
df_transtest.to_csv(f'{DAT_DIR}/test_data.dat', sep=' ', index=False)
#Creating X_train and y_train from train_data.dat
X_train_, y_train_ = load_data(f'{DAT_DIR}/train_data.dat', target_variable = dataset.goal_column)

#Parameters for ProfLogit (can be adapted to experiment, these are the values used for the thesis)
pfl_pd = ProfLogitCS(
    rga_kws={
        'niter': 120,
        'disp': True,
        'random_state': 42,
        'niter_diff': 30
    },
    reg_kws={
        'lambda': 0.001,
    }
)
#Train ProfLogit
pfl_pd.fit(X_train_, y_train_)

#Show best so far solution of ProfLogit
bestsofar= pfl_pd.rga.fx_best
print("The best-so-far solution of ProfLogit are: ")
print(bestsofar)

plt.plot(bestsofar, label='ProfLogit RGA DCCC')
plt.ylabel('Fitness value')
plt.xlabel('Number of iterations')
plt.legend()
plt.show()
# Creating X_test and y_test from test_data.dat
X_test, y_test = load_data(f'{DAT_DIR}/test_data.dat', target_variable= dataset.goal_column)

### Predicted probabilities for defaulters
y_score_= pfl_pd.predict_proba(X_test)

# EMPCS score
empc1 = pfl_pd.score(X_test, y_test.flatten())
print("The EMPCS value of ProfLogit is: ")
print(empc1)
### AUC
auc = metrics.roc_auc_score(y_test, y_score_)
print("The AUC value of ProfLogit is: ")
print(auc)
### H-measure
H = h_score(y_test, y_score_)
print("The H-measure of ProfLogit is: ")
print(H)