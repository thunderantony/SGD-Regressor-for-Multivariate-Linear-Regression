# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import libraries-Load necessary Python libraries.
2. Load Data-Read the dataset containing house details.
3. Preprocess Data-Clean and solit the data into training and testing sets.
4. Select Features & Target-Choose input variables(features) and output variables(house price,occupants).
5. Train Mode-Use SGDRegressor() to train the model.
6. Make Predictions-Use the model to predict house price and occupants.
7. Evaluate Performance-Check accuracy using error metrics.
8.Improve Model-Tune settings for better accuracy.
## Program:
```py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/ad791d97-12e6-4627-82e6-0b1d7f85a704)
```
X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()
```
![image](https://github.com/user-attachments/assets/e10112c5-3b59-4a1d-a9f2-394188f12161)
```
Y = df[['AveOccup','HousingPrice']]
Y.info()
```
![image](https://github.com/user-attachments/assets/a4949cf4-ca8a-429d-9aec-21773f15c834)
```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)
```
![image](https://github.com/user-attachments/assets/a584154f-0dec-44c7-aea1-000950ef4e38)

```
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
```
![image](https://github.com/user-attachments/assets/0a0eadaf-3286-4a3a-87dc-b69939697142)
```
print("\nPredictions:\n", Y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/24402f39-33c8-4877-998b-f7362a9a12c3)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
