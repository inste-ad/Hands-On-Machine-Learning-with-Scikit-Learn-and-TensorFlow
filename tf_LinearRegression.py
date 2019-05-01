#%%
import tensorflow as tf 
import numpy as np
from sklearn.datasets import fetch_california_housing
import os 
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta.eval())
#%%
############梯度下降法#######

n_epochs = 1000
learning_rate = 0.0001

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.assign(theta, theta - learning_rate * gradients)
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())
            sess.run(training_op)

    best_theta = theta.eval()
    print(theta.eval())
#%%[markdown]
## 模型的简化
### 求偏导
#  上面的方法是手算积分表达式。tensoflow最好的地方是提供了差分器。不用自己手算表达式。
# gradients = tf.gradients(mse, [theta])[0]

#%%
gradients = tf.gradients(mse, [theta])
 


#%%[markdown]
# 可以对mse求关于theta的偏导
### 求优化
# 上面采用了梯度下降的方法，在tensorflow里可以使用tf.train 里的方法。
#%%
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)


#%%[markdown]
## Mini-batch
# placeholder 可以看作是空的Variable，后面会被填补。这样可以拿来做mini-batch 不断填补数据进去。使用placeholder 的时候，一定要确保往其中填补了数据。
# 只需要做将X，y设为placeholder点。在run的时候再填补数据。


#%%
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

batch_size = 100

n_batches = int(np.ceil(m / batch_size))

saver = tf.train.Saver()
def fetch_batch(epoch, batch_index, batch_size):
    X_batch = housing_data_plus_bias[batch_index*batch_size:(batch_index+1)*batch_size,:]
    y_batch = housing.target[batch_index*batch_size:(batch_index+1)*batch_size].reshape(-1, 1)
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch%100 == 0:
            print ("Epoch:",epoch)
            save_path = saver.save(sess, "./tmp_model/LinerRegression/my_model.ckpt")
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "./tmp_model/LinerRegression/my_model_final.ckpt")
   
