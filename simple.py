import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

number_of_classes = 2
learning_rate = 0.01
num_steps = 1000

x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
y = tf.placeholder(dtype=tf.float32, shape=[None, number_of_classes], name="y")

w = tf.Variable(tf.random_normal([1, number_of_classes]), name="w")
b = tf.Variable(tf.random_normal([number_of_classes]), name="b")

logits = tf.add(tf.matmul(x, w), b)
# print_logits = tf.Print(logits, [logits], "message")

# prediction = tf.nn.softmax(print_logits)
prediction = tf.nn.softmax(logits)

# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=print_logits, labels=y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="loss")

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op = optimizer.minimize(loss_op)

x_input = np.linspace(0, 100, 99).reshape(-1, 1)
# intermediate = np.random.randn(x_input.shape[0]).reshape(-1, 1)
# y_input = x_input  + intermediate
# y_input = (y_input > 50).squeeze()
y_input = x_input > 50


# def make_it_hot(input, number_of_classes):
#     result = np.zeros((input.shape[0], number_of_classes))
#     for i, label in enumerate(input):
#         result[i, label] = 1
#     return result

def make_it_hot(input, number_of_classes):
    return np.eye(number_of_classes)[input]


plt.ion()
plt.scatter(x_input, y_input)
plt.show()

on_hot_y = make_it_hot(y_input.astype(int), 2)

sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
sess.run(tf.global_variables_initializer())

for step in range(1, num_steps + 1):
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: x_input, y: on_hot_y})
    print(accuracy_val)

result = sess.run(prediction, feed_dict={x: [[2], [53], [1], [3], [5], [83], [49], [50], [48]]})
print([np.argmax(i) for i in result])




