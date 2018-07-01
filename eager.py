import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

a = tf.constant([[1, 2], [4, 5]])

print(tf.matmul(a, a))

def input_fn():
    return tf.contrib.data.make_csv_dataset('abcnews-date-text.csv', 32, num_epochs=10)


for batch in input_fn():
    print(batch)
    break

a = tf.constant([[1, 2], [3, 4]])

print(tf.matmul(a, a))
