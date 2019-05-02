#%%
import numpy as np 
from sklearn import datasets
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#
# TODO    要重新看，问题很多
#
#%%[markdown]
# ridge regression 和 svm 都是对数据的scale 很敏感的。StandardScaler 方法非常实用。
# 关于StandardScaler 可以另外写个笔记做一下比较。
# svm 用超参数C对margin进行控制，越小的C 越包容错误，越不容易over-fitting
# 对于现行可分的数据，我们直接使用linearSVC函数
# 也可以使用SVC(kernel="linear", C=1) 
#%%
iris = datasets.load_iris()
X = iris["data"][:, (2, 3)] # petal length, petal width
y = (iris["target"] == 2).astype(np.float64) # Iris-Virginica

svm_clf = Pipeline((
("scaler", StandardScaler()),
("linear_svc", svm.LinearSVC(C=1, loss="hinge")),
))

svm_clf.fit(X, y)
svm_clf.predict([[5.5, 1.7]])

#%%[markdown]
#The LinearSVC class regularizes the bias term, so you should center the training set first by subtracting its mean. This is automatic if you scale the data using the StandardScaler. Moreover, make sure you set the loss hyperparameter to "hinge", as it is not the default value.
#  Finally, for better performance you should set the dual hyperparameter to False, unless there are more features than training instances (we will discuss duality later in the chapter).