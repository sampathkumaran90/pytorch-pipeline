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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def run_pipeline(input_options):

    trainset = torchvision.datasets.CIFAR10(root=input_options['output'], train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=input_options['output'], train=False,
                                       download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def PrintOptions(options):
    for a in options.items():
        print(a)

def run_pipeline_component(options):
    print("Running data prep job from container")
    
    logging.getLogger().setLevel(logging.INFO)
    
    PrintOptions(options)

    run_pipeline(
        options
    )