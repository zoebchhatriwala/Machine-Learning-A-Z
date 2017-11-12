   # Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
                
# Regressor
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
regressor.fit(x, y)
                

# Visualising (lower resolution)
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising (higher resolution)
grid_x = np.arange(min(x), max(x), 0.1)
grid_x = grid_x.reshape((len(grid_x), 1))
plt.scatter(x, y, color = 'red')
plt.plot(grid_x, regressor.predict(grid_x), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#for 6.5
print(regressor.predict(6.5))

  