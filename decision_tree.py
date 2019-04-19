#%%
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()

X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

#%% [markdown]
# 可以用export graphviz函数对
# **因为不了ipython怎么设置运行文件的，储存文件的路径有一定的问题。所以使用chdir函数进行改变。 os.path.join是拼接文件名**
#%%
from sklearn.tree import export_graphviz
import os
b = os.path.join(os.getcwd(), 'G:\kaggle-learning\Hands-On-Machine-Learning-with-Scikit-Learn-and-TensorFlow')
os.chdir(b)

export_graphviz(
tree_clf,
out_file="iris_tree.dot",
feature_names=iris.feature_names[2:],
class_names=iris.target_names,
rounded=True,
filled=True
)

#%%[markdown]
# 决策树非常容易overfiting。 对有参数的模型来说，对参数进行限制可以降低过拟合的风险，而决策树属于无参数模型。是由max_depth最大树深度这个超参数控制的。
# 其他参数诸如min_samples_split，min_samples_leaf，min_samples_leaf等都可以起到一定的作用