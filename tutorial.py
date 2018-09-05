import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


w = 0.1
b = 0.3

# x_data = np.random.rand(100).astype(np.float32)
x_data = np.linspace(-1, 1, num=100).astype(np.float32)
y_data = w * x_data + b

# "tf.zeros([1])" is same as "tf.zeros(1)"
est_w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
est_b = tf.Variable(tf.zeros([1]))
est_y = est_w * x_data + est_b

# "reduce" means "reduce of tensor_rank."
loss = tf.reduce_mean(tf.square(est_y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# All of tensorflow-variables must be initialized.
init = tf.global_variables_initializer()



# "with" syntax can automatically close.
# sess = tf.Session()
# sess.run(init)
# (something disposal)
# sess.close()
with tf.Session() as sess:
    sess.run(init)
    for step in range(100):
        sess.run(train)
        # if step % 20 == 0:
        print(step, sess.run(est_w), sess.run(est_b))
        plt.scatter(step, sess.run(loss), c='black', s=10)

plt.show()




