# Support Vector Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)
                
#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)

pred_y = regressor.predict(X)

pred_y = sc_y.inverse_transform(pred_y)
                          
#visualize
plt.scatter(x, y, color='red')
plt.plot(x, pred_y, color = 'blue')
plt.title('Trurth or Bluff')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# for 6.5
print(sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]])))))