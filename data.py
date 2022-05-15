from collections import defaultdict
from sklearn.datasets import load_iris
import random


def save(datas, suffix):
    with open(f"iris{suffix}.csv", "w") as fw:
        for target, target_datas in datas.items():
            for data in target_datas:
                data = ",".join([f"{_}" for _ in data])
                fw.write(f"{data},{target}\n")


def main():
    iris = load_iris()
    datas = defaultdict(list)
    for index, data in enumerate(iris["data"]):
        datas[iris["target"][index]].append(data)
    save(datas, "")
    train_datas = {}
    test_datas = {}
    for target, target_datas in datas.items():
        random.shuffle(target_datas)
        train_datas[target] = target_datas[:40]
        test_datas[target] = target_datas[40:]
    save(train_datas, "_train")
    save(test_datas, "_test")


if __name__ == "__main__":
    main()
