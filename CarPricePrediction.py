import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv('car data.csv')
print(car_dataset.head())

label_encoder = LabelEncoder()
car_dataset['Fuel_Type'] = label_encoder.fit_transform(car_dataset['Fuel_Type'])
car_dataset['Transmission'] = label_encoder.fit_transform(car_dataset['Transmission'])
car_dataset['Seller_Type'] = label_encoder.fit_transform(car_dataset['Seller_Type'])

X = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
Y = car_dataset['Selling_Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)

# lin_reg_model = LinearRegression()
# lin_reg_model.fit(X_train,Y_train)


# training_data_prediction = lin_reg_model.predict(X_train)
# error_score = metrics.r2_score(Y_train, training_data_prediction)
# print("R squared Error : ", error_score)


lasso_model = Lasso()
lasso_model.fit(X_train, Y_train)
training_data_prediction = lasso_model.predict(X_train)
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)

from joblib import dump, load
dump(lasso_model, 'CarpricePrediction.joblib') 