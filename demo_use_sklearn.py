from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def load_data(path):
    with open(path) as fr:
        lines = fr.readlines()
    features = []
    labels = []
    for line in lines:
        line = line.strip().split(",")
        features.append(list(map(float, line[2:4])))
        labels.append(int(line[4]))
    return features, labels


def main():
    train_features, train_labels = load_data("iris_train.csv")
    test_features, test_labels = load_data("iris_test.csv")
    sc = StandardScaler()
    sc.fit(train_features)
    train_features_std = sc.fit_transform(train_features)
    test_features_std = sc.fit_transform(test_features)

    p = Perceptron(max_iter=100, random_state=0)
    p.fit(train_features_std, train_labels)
    test_predicts = p.predict(test_features_std)
    score = accuracy_score(test_labels, test_predicts)
    print(score)


if __name__ == "__main__":
    main()
