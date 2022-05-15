import random
import math
from sklearn.metrics import accuracy_score


class Perceptron(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = random.randint(0, len(labels) - 1)
            x = list(features[index])
            x.append(1.0)
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


def load_data(path, select_labels):
    with open(path) as fr:
        lines = fr.readlines()
    features = []
    labels = []
    for line in lines:
        line = line.strip().split(",")
        label = int(line[4])
        if label in select_labels:
            features.append(list(map(float, line[2:4])))
            labels.append(label)
    return features, labels


def standard(features):
    mean = [0.0 for _ in range(len(features[0]))]
    std = [0.0 for _ in range(len(features[0]))]
    for feature in features:
        for dim in range(len(feature)):
            mean[dim] += feature[dim]
            std[dim] += feature[dim] * feature[dim]
    mean = [_ / len(features) for _ in mean]
    std = [math.sqrt(__ / len(features) - mean[_] * mean[_]) for _, __ in enumerate(std)]
    new_features = [[(feature[dim] - mean[dim]) / std[dim] for dim in range(len(feature))] for feature in features]
    return new_features


def main():
    select_labels = (0, 1)
    train_features, train_labels = load_data("iris_train.csv", select_labels=select_labels)
    test_features, test_labels = load_data("iris_test.csv", select_labels=select_labels)
    train_features = standard(train_features)
    test_features = standard(test_features)

    p = Perceptron()
    p.train(train_features, train_labels)

    test_predict = p.predict(test_features)
    score = accuracy_score(test_labels, test_predict)
    print(score)


if __name__ == "__main__":
    main()
