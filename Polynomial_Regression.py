
#%% [markdown]
## Polynomial Regression
#  对于非线性模型，线性回归模型就不适用了。SVM，NN等模型处理非线性模型有很好的效果.对于简单的不能不能用一条直线表达的，我们可以用多项式回归。
# 注意下面式子对X进行了排序，因为X是随机产生的点，是无序的。
#%%
import numpy as np
import sklearn
import matplotlib.pyplot as plt 

m = 100
X = 6 * np.random.rand(m, 1) - 3
X = np.sort(X, axis = 0) #这里的排序因为是列排序，所以用np.sort，写axis 参数。 也可以是用 X.sort(axis = 0)
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X,y)
plt.show()

#%% [markdown]
# 使用PolynomialFeatures 来转换我们的training data，令degree = 2 ，意为二次函数。
#%%
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0],X_poly[0]
#%%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression().fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

#%%
y_pre = lin_reg.predict(X_poly)
plt.plot(X, y_pre)
plt.scatter(X, y)
plt.show()