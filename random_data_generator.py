import numpy as np
import matplotlib.pyplot as plt


def generate_random_features(number_of_features, number_of_classes, sample_size):
    class_size = sample_size // number_of_classes
    x = []
    y = []
    for i in range(number_of_classes):
        j = i * 2
        mean = np.random.randint(0, 50, number_of_features)
        x_class = np.random.multivariate_normal(mean, cov=.1 * np.eye(number_of_features), size=(class_size,))
        y_class = np.full((class_size, 1), i)
        x.append(x_class)
        y.append(y_class)
    # hot = one_hot(np.vstack(y), number_of_classes)
    return np.vstack(x), np.vstack(y).squeeze()

# def one_hot(input, number_of_classes):
#     return np.eye(number_of_classes)[input].squeeze()

if __name__ == '__main__':
    plt.ion()
    x, y = generate_random_features(2, 20, 1000)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
