import tensorflow as tf
import numpy as np

x1 = tf.Variable(1.0)
x2 = tf.Variable(3.0)
y1 = x1**2 + x2 - 1
y2 = x2**3 + x1 - 1

ys = tf.constant(5, dtype=tf.float32)
ys2 = tf.constant(2, dtype=tf.float32)
ys3 = tf.constant(2, dtype=tf.float32)

# sum of ys w.r.t x in xs i.e., every column is a sum of derivatives of all ys with respect with x
grad = tf.gradients([y1, y2], [x1, x2], grad_ys=[1.0, 1.0])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    grad_value = sess.run(grad)
    print(grad_value)


# X = tf.Variable(tf.random_normal([3, 3]))
# X_sum = tf.reduce_sum(X)
# y = X_sum**2 + X_sum - 1
#
# grad = tf.gradients(y, X)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     print(X.eval())
#     print(X_sum.eval())
#     print(y.eval())
#     grad_value = sess.run(grad)
#     print(grad_value)
