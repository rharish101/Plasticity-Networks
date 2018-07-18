#!/usr/bin/env python
import numpy as np
import json
from scipy.ndimage import imread

LOCATION = "/mnt/Data/PascalVOC/"
data = {}

train_file = open(LOCATION + "TrainVal/pascal_trainval.json", "r")
data["train"] = json.load(train_file)
train_file.close()

test_file = open(LOCATION + "Test/pascal_test.json", "r")
data["test"] = json.load(test_file)
test_file.close()


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
        size_arr = np.array(
            list(zip(item["size"][:2], item["size"][:2]))
        ).flatten()
        # Image target consists of normalized bounding boxes and the label in
        # the form: (x_max, x_min, y_max, y_min, label)
        yield imread(LOCATION + folder + "JPEGImages/" + item["filename"]), [
            list(np.array(bbox) / size_arr) + [labels_list.index(label)]
            for label, bbox in zip(item["labels"], item["bndbox"])
        ]
