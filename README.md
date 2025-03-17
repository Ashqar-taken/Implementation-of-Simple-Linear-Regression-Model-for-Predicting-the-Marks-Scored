# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.
2. Assigning hours to X and scores to y.
3. Plot using the Scatter plot.
4. Use mse, mae, rmse formulae to find the values and display.
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Ashqar Ahamed S.T
RegisterNumber:  212224240018
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'C:\College\SEM 2\Machine Learning\Exp2\student_scores.csv')

print(df.head())
print(df.tail())

X=df.iloc[:,:-1].values
print("X vlaues: ")
print(X)
y=df.iloc[:,1].values
print("Y values")
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
print("The predicted values: ")
print(y_pred)
print("Test values: ")
print(y_test)

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse = np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

## Displaying the head and tail values:
![head and tail](https://github.com/user-attachments/assets/c183f264-ef7e-42e9-a2eb-493de0bfadc5)

## Comparing Datasets:
![comparing dataset](https://github.com/user-attachments/assets/e49f9e09-e798-4a36-bcdc-5cd41b210e72)

## Predicted values:
![predicted values](https://github.com/user-attachments/assets/d2b69572-b9cd-48f3-8898-01dc62e33d40)

## Test values:
![test values](https://github.com/user-attachments/assets/5d921381-8713-486d-a3a3-f7066452795a)

## Error 
![Performance](https://github.com/user-attachments/assets/fdfc444b-8a9d-4020-9a46-9092494a233e)

## Graph for training set:
![Figure_1](https://github.com/user-attachments/assets/9f1b105c-e980-4a8b-a12d-b84547dd83c6)

## Graph for testing set:
![Figure_2](https://github.com/user-attachments/assets/c9ba2647-63c4-430d-aaa0-eaa0913d7de0)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
