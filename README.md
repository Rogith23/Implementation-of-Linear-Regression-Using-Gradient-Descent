# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. open juypter notebook
2. write the code
3. get the output 
4. put the output in github and commitchanges

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ROGITH J
RegisterNumber:  212224040280
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term
    X = np.c_[np.ones(len(X1)), X1]

    # Initialize theta with zeros
    theta = np.zeros((X.shape[1], 1)).reshape(-1, 1)

    # Perform gradient descent
    for _ in range(num_iters):
        # Calculate predictions
        predictions = X.dot(theta).reshape(-1, 1)

        # Calculate errors
        errors = (predictions - y).reshape(-1, 1)

        # Update theta using gradient descent
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)

    return theta


data = pd.read_csv(r"C:\Users\admin\Downloads\50_Startups.csv", header=None)
print (data.head())
# Assuming the last column is your target variable 'y' and the preceding columns are your features
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform (X1)
Y1_Scaled = scaler.fit_transform(y)
print('Name:ROGITH J')
print('Register No.:212224040280')
print(X1_Scaled)
print(Y1_Scaled)
# Learn model parameters
theta = linear_regression (X1_Scaled, Y1_Scaled)
# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
*/
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="860" height="414" alt="Screenshot 2025-09-01 183002" src="https://github.com/user-attachments/assets/2e2af347-f7fa-41ee-8adc-c6dc99853c6e" />
<img width="765" height="308" alt="Screenshot 2025-09-01 183011" src="https://github.com/user-attachments/assets/34d824b1-b2b0-4ebe-9da9-9121b0e79efe" />
<img width="483" height="126" alt="Screenshot 2025-09-01 183702" src="https://github.com/user-attachments/assets/d0804fc9-b000-41de-b405-2b6cda265055" />







## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
