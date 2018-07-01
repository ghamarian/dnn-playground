import tensorflow as tf

from tensorflow.python import debug as tf_debug

# iterator = tf.data.Dataset.from_tensors([[1, 2, 3], [4, 5, 6]]).make_one_shot_iterator()
# iterator2 = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4]).make_one_shot_iterator()

prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
                    [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
                    [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

# dataset = tf.data.Dataset.from_tensor_slices(([['a'], ['b']], ([1, 2], ['k', 'l'])))
dataset = tf.data.Dataset.from_tensor_slices(([['a'], ['b']], ([1, 2], ['k', 'l'])))

# dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
my_iterator = dataset.make_one_shot_iterator()

with tf.Session() as sess:
    print(sess.run(my_iterator.get_next()))
    print(sess.run(my_iterator.get_next()))

# prediction_input = [[5.9, 3.0, 4.2, 1.5],  # -> 1, Iris Versicolor
#                     [6.9, 3.1, 5.4, 2.1],  # -> 2, Iris Virginica
#                     [5.1, 3.3, 1.7, 0.5]]  # -> 0, Iris Sentosa

# prediction_input = [[5.9, 'a', 4.2, 1.5],  # -> 1, Iris Versicolor
#                     [6.9, 'a', 5.4, 2.1],  # -> 2, Iris Virginica
#                     [5.1, 'a', 1.7, 0.5]]  # -> 0, Iris Sentosa
# feature_names = [
#     'SepalLength',
#     'SepalWidth',
#     'PetalLength',
#     'PetalWidth']
#
# def new_input_fn():
#    def decode(x):
#        x = tf.split(x, 4) # Need to split into our 4 features
#        # When predicting, we don't need (or have) any labels
#        return dict(zip(feature_names, x)) # Then build a dict from them
#
#    # The from_tensor_slices function will use a memory structure as input
#    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
#    dataset = dataset.map(decode)
#    iterator = dataset.make_one_shot_iterator()
#    next_feature_batch = iterator.get_next()
#    return next_feature_batch, None # In prediction, we have no labels
#
# a, _ = new_input_fn()
#
# # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
#
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(a))
#     print(sess.run(a))
#
