
#To read the ecxal sheet
import pandas as pd

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

#Linear Regression model
from sklearn.linear_model import LinearRegression

#Functions to Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score

#To create graphs
import matplotlib.pyplot as plt

#To save the model
import joblib

# Load the dataset of Housing Prices dataset
dataset = pd.read_csv('california_housing.csv')

# x is features (independent variables) and y is a target (dependent variable)
X = dataset[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']] 
y = dataset['PRICE']

# Split the data into training and testing sets
#(X_train) : x vraiables for training
#(y_train) : y vraiables for training
#(X_test)  : X vraiables for evaluation
#(y_test)  : y vraiables for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create the model
model = LinearRegression()

#Training the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.5f}')
print(f'R-squared: {r2:.2f}')

# Visualize predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

# Save the model to a file
joblib.dump(model, 'multi_Linear_Regression_model.joblib')
