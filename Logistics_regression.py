#%%[markdown]
## Logistic Regression

#%%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int)

#%%[markdown]
# 代码中用了一个小trick。x\[y==0\].
# 其实是利用y的排序对X进行挑选。y==0，print之后其实是\[True True False ...\]，那么X[y==0, 0],就X的前三个元素而言，分别代表返回（包含），返回（包含），不返回（不包含）；与之类似对于“iris["data"][:, (2, 3)]”这个写法，左边的那个"："代表的行是[True，True, ... ,True]，全是true；
#%%
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")

#%% [markdown]
## Softmax Regression
#  The Logistic Regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers (as discussed in Chapter 3). This is called Softmax Regression, or Multinomial Logistic Regression.



#%%
X, y = datasets.load_iris(return_X_y=True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial').fit(X, y)
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :]) 
clf.score(X, y)




