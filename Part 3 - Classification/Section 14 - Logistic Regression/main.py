# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
                
                
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# Regressor
from sklearn.linear_model import LogisticRegression
classifier =  LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

#Predicting
y_pred = classifier.predict(x_test)

# Confusing Metric
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the Test set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#visual manual - pred result
x_up = sc_X.inverse_transform(x_test)
y_up = y_pred
bought1 = np.array([])
bought2 = np.array([])
notbought1 = np.array([])
notbought2 = np.array([])

for i in range(0, len(x_up)):
    if y_up[i] == 1:
        bought1 = np.append(bought1, x_up[i][0])
        bought2 = np.append(bought2, x_up[i][1])
    else:
        notbought1 = np.append(notbought1, x_up[i][0])
        notbought2 = np.append(notbought2, x_up[i][1])
plt.scatter(bought1, bought2, color='green')
plt.scatter(notbought1, notbought2, color='red')
plt.title('Logistic Regression (Predicted)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()

#visual manual - test results
x_up = sc_X.inverse_transform(x_test)
y_up = y_test
bought1 = np.array([])
bought2 = np.array([])
notbought1 = np.array([])
notbought2 = np.array([])

for i in range(0, len(x_up)):
    if y_up[i] == 1:
        bought1 = np.append(bought1, x_up[i][0])
        bought2 = np.append(bought2, x_up[i][1])
    else:
        notbought1 = np.append(notbought1, x_up[i][0])
        notbought2 = np.append(notbought2, x_up[i][1])
plt.scatter(bought1, bought2, color='green')
plt.scatter(notbought1, notbought2, color='red')
plt.title('Logistic Regression (Original Result)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.show()


    
