#%%
import os
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

n_inputs = 28*28 # MNIST 28*28 pixel
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

#%%[markdown]
# 由于由于不知道每次batch（图片）是多少，所以我们用shape（none，28）来规定placeholder的形状。
#%%
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name ="y" )
# 由于y是分类标签，所以用int

#%%
def neuron_layer(X, n_neurons, name, activation=None):
	with tf.name_scope(name):
		n_inputs = int(X.get_shape()[1])
		stddev = 2 / np.sqrt(n_inputs)
		init = tf.truncated_normal(shape = (n_inputs, n_neurons), stddev = stddev)
		W = tf.Variable(init, name="weights")
		b = tf.Variable(tf.zeros([n_neurons]), name="biases")
		z = tf.matmul(X, W) + b
		if activation=="relu":
			return tf.nn.relu(z)
		else:
			return z
#%%[markdown]
# 首先用函数传进来的name定义了一个命名空间(一般是layer名字)。</br>
# 得到输入的第二列（第一列作为示例）</br>
# 接下来的三行代码都是在初始化W。用的是truncted_normal 高斯分布(确保不会有大的高斯值）。随机生成</br>
# 接下来就是初始化b
# 最后用Xw+b构建一个子图。用RELU 为激活函数进行返回。</br>
# 接下来就用neuron_layer函数来构建DNN
#%%
with tf.name_scope("dnn"):
	hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
	hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
	logits = neuron_layer(hidden2, n_outputs, "outputs")
#%%[markdown]
# 上面的也可以是用集成好的fully_connected 来做。没有必要自己定义neruon_layer函数。但是下面这个包是属于TensorFlow中的实验性包，在未来有可能删除
# 现在已经出现TensorFlow 2.0了，并且集成了keras，其他非keras的包都不建议使用。

'''
from tensorflow.contrib.layers import fully_connected
with tf.name_scope("dnn"):
	hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
	hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
	logits = fully_connected(hidden2, n_outputs, scope="outputs",
	activation_fn=None)
'''
#%%
# Loss Function: entropy
with tf.name_scope("loss"):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(xentropy, name="loss")


# update
learning_rate = 0.01

with tf.name_scope("train"):
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
	correct = tf.nn.in_top_k(logits, y, 1)
	accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

############运行#######

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST")

saver = tf.train.Saver()
init = tf.global_variables_initializer()
n_epochs = 100
batch_size = 50


test = np.ones(n_epochs)
train = np.ones(n_epochs)


with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		for iteration in range(mnist.train.num_examples // batch_size):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
		acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
		acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
		y: mnist.test.labels})
		train[epoch] = acc_train
		test[epoch] = acc_test
		print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
		
	save_path = saver.save(sess, "./tmp_model/fully_connnected/my_model_final.ckpt")
	#save_path = saver.save(sess, "./my_model_final.ckpt")
plt.figure()
plt.plot( range(n_epochs),train,"g")
plt.plot( range(n_epochs),test,"r")
plt.show()

# 下面是要再次启用储存模型的代码
""" 
with tf.Session() as sess:
	saver.restore(sess, "./my_model_final.ckpt")
	X_new_scaled = [...] # some new images (scaled from 0 to 1)
	Z = logits.eval(feed_dict={X: X_new_scaled})
	y_pred = np.argmax(Z, axis=1)
 """