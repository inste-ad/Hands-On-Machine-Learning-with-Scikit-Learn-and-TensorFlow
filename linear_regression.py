#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import LinearRegression
#class sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
from sklearn.metrics import mean_squared_error, r2_score



alpha =6
X = np.linspace(10,20,30).reshape(-1,1)
y_1 = 4+3*X + alpha*np.random.rand(30,1)

#%%
lin_reg = LinearRegression()
lin_reg.fit(X, y_1)


y_predict=lin_reg.predict(X)

# The coefficients
print('Coefficients: \n',  lin_reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_predict, y_1))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_predict, y_1))

plt.scatter(X, y_1)
plt.plot(X, y_predict,'r')

plt.show()

'''
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y_predict = 1 * x_0 + 2 * x_1 + 3
y_predict = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y_predict)
reg.score(X, y_predict)

reg.coef_

reg.intercept_ 

reg.predict(np.array([[3, 5]]))
'''