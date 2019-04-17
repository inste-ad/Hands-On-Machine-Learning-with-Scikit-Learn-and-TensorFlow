#%%
import numpy as np 
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()

X = iris.data[:, 2:] # petal length and width
y = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)
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
