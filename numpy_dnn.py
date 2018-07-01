import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def softmax(v):
    s = np.sum(np.exp(v), axis=1)[:, None]
    return np.exp(v) / s


# w1 . a[0] --> relu --> w2 . a[1] --> relu --> w3 . a[2] --> softmax --> crossentropy

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(np.sum(targets * np.log(predictions))) / N
    return ce


def derivative_cross(yhat, y):
    s = softmax(yhat)
    return s - y


targets = np.array([[0.0, 0, 1], [0, 0, 1], [1, 0, 0]])
yhat = np.array([[0.2, 0.1, 0.7], [61, 24, 58], [78, 12, 11]])

tf_target = tf.get_variable('targets', shape=targets.shape)
tf_yhat = tf.get_variable('yhat', shape=yhat.shape)

tf_target.load(targets)
tf_yhat.load(yhat)

with tf.GradientTape() as tape:
    loss_value = tf.nn.softmax_cross_entropy_with_logits(labels=tf_target, logits=tf_yhat)

grads = tape.gradient(loss_value, [tf_yhat])
print('tensorflow calculations', grads)

amir_grads = derivative_cross(yhat, targets)
print('amirs calculations', amir_grads)

np.testing.assert_allclose(amir_grads, grads[0])


def grad_check(yhat, y, epsilon=1e-7):
    yhat_reshaped = yhat.reshape(-1).squeeze()
    gradapprox = np.zeros(yhat_reshaped.shape).squeeze()

    for i, v in enumerate(yhat_reshaped):
        k_p = np.copy(yhat_reshaped)
        k_p[i] = v + epsilon

        k_m = np.copy(yhat_reshaped)
        k_m[i] = v - epsilon

        k_m = k_m.reshape((-1, 3))
        k_p = k_p.reshape((-1, 3))

        s_p = softmax(k_p)
        s_m = softmax(k_m)

        gradapprox[i] = ((cross_entropy(s_p, y) - cross_entropy(s_m, y)) / (2 * epsilon))

    grad = derivative_cross(yhat, y).reshape(-1).squeeze() / len(yhat)

    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator

    print(difference)


grad_check(yhat, targets)


def train(features, targets):
    nr_features = features.shape[1]
    nr_classes = len(targets.unique())

    w1 = np.random.uniform(-1, 1, (nr_features, 10))
    w2 = np.random.uniform(-1, 1, (10, 5))
    output_w = np.random.uniform(-1, 1, (5, nr_classes))

    for i in range(200):
        layer1 = np.max(0, np.dot(features, w1))
        layer2 = np.max(0, np.dot(layer1, w2))
        output_layer = np.dot(layer2, output_w)

        predictions = softmax(output_layer)
        print(cross_entropy(predictions, targets))
