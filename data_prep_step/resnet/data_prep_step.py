import sys, argparse, logging
import os
import torch.utils.data
from PIL import Image
import json
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import webdataset as wds
from pathlib import Path
from itertools import islice
from sklearn.model_selection import train_test_split


def run_pipeline(input_options):

    trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True)

    Path(input_options["output"] + "/train").mkdir(parents=True, exist_ok=True)
    Path(input_options["output"] + "/val").mkdir(parents=True, exist_ok=True)
    Path(input_options["output"] + "/test").mkdir(parents=True, exist_ok=True)

    random_seed = 25
    y = trainset.targets
    trainset, valset, y_train, y_val = train_test_split(
        trainset, y, stratify=y, shuffle=True, test_size=0.2, random_state=random_seed
    )

    for name in [(trainset, "train"), (valset, "val"), (testset, "test")]:
        with wds.ShardWriter(
            input_options["output"] + "/" + str(name[1]) + "/" +  str(name[1]) + "-%d.tar", maxcount=1000
        ) as sink:
            for index, (image, cls) in enumerate(name[0]):
                sink.write({"__key__": "%06d" % index, "ppm": image, "cls": cls})


def PrintOptions(options):
    for a in options.items():
        print(a)


def run_pipeline_component(options):
    print("Running data prep job from container")

    logging.getLogger().setLevel(logging.INFO)

    PrintOptions(options)

    run_pipeline(options)

# if __name__ == "__main__":
#     run_pipeline({"output": "test/webdataset"})