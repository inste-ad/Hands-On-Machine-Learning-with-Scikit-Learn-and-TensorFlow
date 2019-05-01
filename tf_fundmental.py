#%%
import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

#%%
sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)

#%%
with tf.Session as sess:
	x.initializer.run()
	y.initializer.run()
	result = f.eval()

#%%[markdown]
# initializer 也可以使用全局初始化，这样就可以只用初始化一次
#%%
init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
	init.run() # actually initialize all the variables
	result = f.eval()


#%%[markdown]
# 这个例子中 计算y的时候，tf检测到y是依赖于w的（构建的图模型可以看出）所以tf计算的时候是先计算w，后计算x，最后y。相当于是图中的流。
# 当再次求z的时候，会再计算w和x的值

#%%
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3
with tf.Session() as sess:
	print(y.eval()) # 10
	print(z.eval()) # 15

#%% [markdown]
# 如果想要计算同时y和z，而不是像上边那样计算两次，那么最好用一个图运行两个计算，如下例。

#%%
with tf.Session() as sess:
	y_val, z_val = sess.run([y, z])
	print(y_val) # 10
	print(z_val) # 15