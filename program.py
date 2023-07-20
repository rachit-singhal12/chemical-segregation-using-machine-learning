#importing the necssary library

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score
from sklearn.linear_model import LogisticRegression

#Reading the training data and testing data
x_train = pd.read_csv('Training Data/Logistic_X_Train.csv')
y_train = pd.read_csv('Training Data/Logistic_Y_Train.csv')
x_test = pd.read_csv('Test Cases/Logistic_X_Test.csv')
sample_data = pd.read_csv('Test Cases/SampleOutput.csv')

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.fit_transform(x_test)


model = LogisticRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred,sample_data)

print(r2_score(sample_data.head(1000),y_pred))
print(mean_absolute_error(sample_data.head(1000),y_pred))
print(accuracy_score(sample_data.head(1000),y_pred))