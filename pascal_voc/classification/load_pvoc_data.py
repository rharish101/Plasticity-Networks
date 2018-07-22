#!/usr/bin/env python
import json
from scipy.ndimage import imread

LOCATION = "/mnt/Data/PascalVOC/"
data = {}

with open(LOCATION + "TrainVal/pascal_trainval.json", "r") as train_file:
    data["train"] = json.load(train_file)
    TRAIN_LENGTH = len(data["train"])

with open(LOCATION + "Test/pascal_test.json", "r") as test_file:
    data["test"] = json.load(test_file)


def load_data(dataset):
    if dataset == "train":
        folder = "TrainVal/"
    elif dataset == "test":
        folder = "Test/"
    else:
        raise (ValueError("datset type must be one of: 'train', 'test'"))
    labels_list = sorted(
        set([label for item in data[dataset] for label in item["labels"]])
    )
    for item in data[dataset]:
        yield imread(LOCATION + folder + "JPEGImages/" + item["filename"]), [
            labels_list.index(label) for label in item["labels"]
        ]
