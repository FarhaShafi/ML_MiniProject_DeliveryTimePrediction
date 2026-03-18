# Import nessesary libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import pickle

import warnings
warnings.filterwarnings('ignore')

# Load Data
df = pd.read_csv('talabat_enhanced_orders.csv')
# df.head()

# Feature Selection
df = df.drop(columns=['Order_ID', 'User_ID', 'Restaurant_ID', 'Driver_ID',
                      'Restaurant_Lat', 'Restaurant_Lon', 'Customer_Lat',
                      'Customer_Lon', 'Driver_Lat', 'Driver_Lon',
                      'Order_Time', 'Delivery_Time'])


# Handling Outliers
outlier_column = ['Delivery_Distance_km']

def remove_outliers_iqr(data, column):
    q1, q2, q3 = np.percentile(data[column], [25, 50, 75])
    # print("q1, q2, q3 : ", q1, q2, q3)
    IQR = q3 - q1
    # print("IQR : ", IQR)
    lower_limit = q1 - (1.5 * IQR)
    upper_limit = q3 + (1.5 * IQR)
    data[column] = np.where(data[column] > upper_limit, upper_limit, data[column]) # Capping the upper limit
    data[column] = np.where(data[column] < lower_limit, lower_limit, data[column]) # Flooring the lower limit

for column in outlier_column:
    remove_outliers_iqr(df, column)

# Feature Engineering
# Label Encoding the Item_Name
le_Item_Name = LabelEncoder()
df['Item_Name'] = le_Item_Name.fit_transform(df['Item_Name'])

with open('Item_Name.pkl', 'wb') as f:
    pickle.dump(le_Item_Name, f)

# Label Encoding the City
le_City = LabelEncoder()
df['City'] = le_City.fit_transform(df['City'])

with open('City.pkl', 'wb') as f:
    pickle.dump(le_City, f)

# Label Encoding the Driver_Vehicle
le_Driver_Vehicle = LabelEncoder()
df['Driver_Vehicle'] = le_Driver_Vehicle.fit_transform(df['Driver_Vehicle'])

with open('Driver_Vehicle.pkl', 'wb') as f:
    pickle.dump(le_Driver_Vehicle, f)

# Ordinal Encoding the Traffic_Level
traffic_level_map = {'High' : 0, 'Low' : 1, 'Medium' : 2}
df['Traffic_Level'] = df['Traffic_Level'].map(traffic_level_map)

# Ordinal Encoding the Payment_Method
payment_method_map = {'Wallet' : 0, 'Credit Card' : 1, 'Cash' : 2}
df['Payment_Method'] = df['Payment_Method'].map(payment_method_map)

# Ordinal Encoding the Order_Status
order_status_map = {'Delivered' : 0, 'In Transit' : 1, 'Cancelled' : 2}
df['Order_Status'] = df['Order_Status'].map(order_status_map)

# One-hot Encoding the Driver_Availability
df = pd.get_dummies(df, columns=['Driver_Availability'], drop_first=True, dtype=int)

# Feature Scaling
# Splitting data into dependent and independent columns
x = df.drop('Delivery_Duration_Minutes', axis = 1)
y = df['Delivery_Duration_Minutes']

# Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Coverting to Dataframe
x = pd.DataFrame(x)

with open('scaling.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# Train a XGB regressor model
model = xgb.XGBRegressor(n_estimators = 20, max_depth = 5)

# Fit the model on the training data
model.fit(X_train, y_train)

# Save the model 
pickle.dump(model, open('model.pkl','wb'))
