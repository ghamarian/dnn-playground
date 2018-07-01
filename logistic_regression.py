import tensorflow as tf
from random_data_generator import generate_random_features

number_of_features = 20
number_of_classes = 20

input = tf.placeholder(tf.float32, shape=(None, number_of_features))
target_input = tf.placeholder(tf.int64, shape=(None,))
target = tf.one_hot(target_input, number_of_classes)

weight = tf.Variable(tf.random_uniform([number_of_features, number_of_classes]), name="weight")
bias = tf.Variable(tf.random_uniform([number_of_classes]), name="bias")

logits = tf.matmul(input, weight) + bias
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

prediction = tf.nn.softmax(logits)
mistakes = tf.not_equal(target_input, tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float64))

x, y = generate_random_features(number_of_features, number_of_classes, 10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _, loss_value = sess.run([optimizer, loss], feed_dict={input: x, target_input: y})
        print(sess.run(error, feed_dict={input: x, target_input: y}))
