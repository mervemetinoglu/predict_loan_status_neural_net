import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import KrediNN as model

import warnings
warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None

ds = pd.read_csv("D:\\ml\\predict_loan_status\\train_kredi_tahmini.csv")


ds_crop = ds.dropna(subset=["Gender", "Married"])     

ds_crop[['Self_Employed']] = ds_crop[['Self_Employed']].replace('No', 0)
ds_crop[['Self_Employed']] = ds_crop[['Self_Employed']].replace('Yes', 1)

ds_crop[['Dependents']] = ds_crop[['Dependents']].replace('0', 0)
ds_crop[['Dependents']] = ds_crop[['Dependents']].replace('1', 1)
ds_crop[['Dependents']] = ds_crop[['Dependents']].replace('2', 2)
ds_crop[['Dependents']] = ds_crop[['Dependents']].replace('3+', 3)

ds_crop[['Loan_Status']] = ds_crop[['Loan_Status']].replace('N', 0)
ds_crop[['Loan_Status']] = ds_crop[['Loan_Status']].replace('Y', 1)


ds_fill = ds_crop.fillna(ds_crop.median())


ds_fill['Dependents'] = ds_fill['Dependents'].astype('int64')
ds_fill['Self_Employed'] = ds_fill['Self_Employed'].astype('int64')
ds_fill['CoapplicantIncome'] = ds_fill['CoapplicantIncome'].astype('int64')
ds_fill['LoanAmount'] = ds_fill['LoanAmount'].astype('int64')
ds_fill['Loan_Amount_Term'] = ds_fill['Loan_Amount_Term'].astype('int64')
ds_fill['Credit_History'] = ds_fill['Credit_History'].astype('int64')




features = ['Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']
ds_x = ds_fill.loc[:, features]
ds_y = ds_fill.loc[:, ['Loan_Status']]

ds_x = StandardScaler().fit_transform(ds_x)


X_train, X_test, y_train, y_test = train_test_split(np.array(ds_x), np.array(ds_y), test_size = 0.4, random_state = 44)
model = model.KrediNN(X_train.shape[1])

model.train(X_train, y_train)

true = 0
false = 0
for i in range(X_test.shape[0]):
    t = model.test(X_test[i])
    if round(t[0]) == y_test[i]:
        true += 1
    else :
        false += 1

print("True predicts: ", true)
print("False predicts: ", false)
print("Predict: ", true/(true + false))



