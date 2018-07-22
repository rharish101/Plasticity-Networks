#!/usr/bin/env python
import numpy as np
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
        # Image target consists of normalized bounding boxes and the label in
        # the form: (x_center, y_center, width, height, label)
        img_target = []
        size_arr = np.array(item["size"][:2] + item["size"][:2])
        for label, bbox in zip(item["labels"], item["bndbox"]):
            new_bbox = [
                (bbox[0] + bbox[1]) / 2,  # bx
                (bbox[2] + bbox[3]) / 2,  # by
                (bbox[0] - bbox[1]) / 2,  # bw
                (bbox[2] - bbox[3]) / 2,  # bh
            ]
            img_target.append(
                list(np.array(new_bbox) / size_arr)
                + [labels_list.index(label)]
            )
        yield imread(
            LOCATION + folder + "JPEGImages/" + item["filename"]
        ), img_target
