import joblib
import pandas as pd

#Load the models to use
MLR = joblib.load('multi_Linear_Regression_model.joblib')
DTR = joblib.load('decision_tree_regressor_model.joblib')
RFR = joblib.load('random_forest_model.joblib')


# x variables to predict the price
MedInc = 8.3252
HouseAge = 41
AveRooms = 6.984126984
AveBedrms = 1.023809524
Population = 322
AveOccup = 2.555555556
Latitude = 37.88
Longitude = -122.23

#to compare
# the actual price is 4.526
price = 4.526

#frame data to give the x variable to ther models
X_test = pd.DataFrame({
    'MedInc': MedInc,
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [AveBedrms],
    'Population': [Population],
    'AveOccup': [AveOccup],
    'Latitude': [Latitude],
    'Longitude': [Longitude]
})


# Make predictions
MLRP = predictions = MLR.predict(X_test)
DTRP = predictions = DTR.predict(X_test)
RFRP = predictions = RFR.predict(X_test)

# Print the predicts values 
print(f"the Aactual price is: {price:.5f}")
print(f"MLR: {MLRP}")
print(f"DTR: {DTRP:}")
print(f"RFR: {RFRP:}")