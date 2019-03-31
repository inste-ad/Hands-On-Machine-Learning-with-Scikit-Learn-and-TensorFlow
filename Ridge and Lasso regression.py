#%%
import numpy as np
import sklearn
import matplotlib.pyplot as plt 


a = 1

alpha =6
X = np.linspace(10,20,30).reshape(-1,1)
y = 4+3*X + alpha*np.random.rand(30,1)



#%%
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha= a , solver="cholesky")
ridge_reg.fit(X,y)


#%%
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha= a)
lasso_reg.fit(X,y)

#%%
import matplotlib.pyplot as plt
def plot(model):
    y_pre = model.predict(X)
    plt.plot(X,y_pre)
    plt.scatter(X, y)
    plt.show()
    

plot(lasso_reg)
plot(ridge_reg)