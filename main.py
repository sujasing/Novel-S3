
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

datasetTrain = pd.read_csv('C:\\Amrita\\WIDS\\widsdatathon2022\\train_processed.csv')
print(datasetTrain.head(5))
print(datasetTrain.describe())

# One-hot encode the data using pandas get_dummies
datasetTrain = pd.get_dummies(datasetTrain)
print(datasetTrain.head(5))

labels = np.array(datasetTrain['site_eui'])

# Remove the labels from the datasetTrain
# axis 1 refers to the columns
datasetTrain = datasetTrain.drop('site_eui', axis=1)

# Saving feature names for later use
datasetTrain_list = list(datasetTrain.columns)

# Convert to numpy array
datasetTrain = np.array(datasetTrain)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(datasetTrain, labels, test_size=0.25, random_state=0)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Instantiate model with 1000 decision trees
# rf = RandomForestRegressor(n_estimators=1000, random_state=42)
rf = RandomForestRegressor(n_estimators=50, random_state=0)
# Train the model on training data
rf.fit(X_train, y_train);

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
#datasetTrain = datasetTrain[rf.predict(X_test)]
#predictions.csv('C:\\Amrita\\WIDS\\widsdatathon2022\\rf_result.csv')
#print(predictions)
# Calculate the absolute errors
#errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'eui.')

# Calculate mean absolute percentage error (MAPE)
#mape = 100 * (errors / y_test)
# Calculate and display accuracy
#accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%.')

