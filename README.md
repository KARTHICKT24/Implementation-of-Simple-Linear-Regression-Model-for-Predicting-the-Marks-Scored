# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import required libraries.

2.Load and explore the dataset.

3.Split dataset into training and testing sets.

4.Train the Simple Linear Regression model on training data.

5.Predict marks using the test data.

6.Evaluate model performance (MAE, MSE, R²).

7.Visualize actual vs. predicted results.

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK KISHORE T
RegisterNumber: 212223220042

```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print("")
print(x)
y=dataset.iloc[:,1].values
print("")
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print("")
print(y_pred)
print("")
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("NAME : KARTHICK KISHORE T")
print("Reg No: 212223220042")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("NAME : KARTHICK KISHORE T")
print("Reg No: 212223220042")
plt.show()

print("NAME : KARTHICK KISHORE T")
print("Reg No: 212223220042")
mse=mean_absolute_error(y_test,y_pred)
print("")
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print("")
print('Mean Absolute Error = ',mae)
print("")
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```
*/


## Output:
## HEAD AND TAIL VALUES
<img width="233" height="312" alt="2 1" src="https://github.com/user-attachments/assets/bc450e23-a51e-4fed-9367-5fe23ba2a8b4" />
## x values
<img width="246" height="563" alt="2 3" src="https://github.com/user-attachments/assets/1b53c0e3-3be0-49c3-99b0-b787a97cb842" />
## y values
<img width="703" height="58" alt="2 4" src="https://github.com/user-attachments/assets/07e676fb-72de-48c0-b644-1490dcc51ffb" />
## x predicted values
<img width="695" height="42" alt="2 5" src="https://github.com/user-attachments/assets/eda751be-a4ef-4908-9c27-1559895d6d93" />
## y predicted values
<img width="331" height="30" alt="2 6" src="https://github.com/user-attachments/assets/0cc05f96-38d5-4e26-bc42-d7cfbdb3e1fc" /> 

<img width="741" height="602" alt="2 7" src="https://github.com/user-attachments/assets/e38b6225-db43-4619-9b74-61011626bee0" />

<img width="737" height="606" alt="2 8" src="https://github.com/user-attachments/assets/d7aa77b3-3d00-41a1-b90c-3d25ec0fb8f5" />

<img width="432" height="172" alt="2 9" src="https://github.com/user-attachments/assets/4ebbaea9-6abe-4736-ac7c-e08690f80a5b" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
