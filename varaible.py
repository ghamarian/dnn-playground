import tensorflow as tf
from tensorflow.python import debug as tf_debug

# with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
#     v = tf.get_variable("v", [1])
# with tf.variable_scope("foo", reuse=True):
#     v1 = tf.get_variable("v", [1])
#
# with tf.name_scope('hidden') as scope:
#   a = tf.constant(5, name='alpha')
#   W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
#   b = tf.Variable(tf.zeros([1]), name='biases')
#
# with tf.Session() as sess:
#     sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Amirs-MacBook-Pro-2.local:7000")
#     sess.run(tf.global_variables_initializer())


v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
assignment = v.assign_add(3)
different = v.assign_add(4)

sess = tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
sess.run(tf.global_variables_initializer())
print(sess.run(different))  # or assignment.op.run(), or assignment.eval()
print(sess.run(assignment))  # or assignment.op.run(), or assignment.eval()
print(sess.run(assignment))  # or assignment.op.run(), or assignment.eval()
print(sess.run(assignment))  # or assignment.op.run(), or assignment.eval()
print(sess.run(v))  # or assignment.op.run(), or assignment.eval()

# v = tf.get_variable("v", shape=(), initializer=tf.zeros_initializer())
# assignment = v.assign_add(1)
#
# with tf.control_dependencies([assignment]):
#   w = v.read_value()  # w is guaranteed to reflect v's value after the
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(w))
#
#                           # assign_add operation.
