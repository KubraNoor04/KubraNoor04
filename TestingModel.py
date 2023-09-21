import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn import metrics
model = joblib.load('CarpricePrediction.joblib')
data_tobeTest = pd.read_csv('TestingDataFrame.csv')
print(data_tobeTest.columns)
# Create new label encoders for categorical columns
label_encoder_fuel_type = LabelEncoder()
label_encoder_seller_type = LabelEncoder()
label_encoder_transmission = LabelEncoder()

# Fit the new label encoders to the categorical columns in the unseen data
data_tobeTest['Fuel_Type'] = label_encoder_fuel_type.fit_transform(data_tobeTest['Fuel_Type'])
data_tobeTest['Seller_Type'] = label_encoder_seller_type.fit_transform(data_tobeTest['Seller_Type'])
data_tobeTest['Transmission'] = label_encoder_transmission.fit_transform(data_tobeTest['Transmission'])


expected_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
test_data = data_tobeTest[expected_columns]

# Apply any necessary data preprocessing (e.g., label encoding)

# Make predictions using the model
predictions = model.predict(test_data)

# Print or save the predictions
print(predictions)
actual_prices = [8.0, 2.0, 6.5, 2.2, 7.2]
error_score = metrics.r2_score(actual_prices, predictions)
print("R squared Error : ", error_score)

