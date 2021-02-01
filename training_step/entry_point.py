import argparse
import os
from os import path
import json
from train import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--train_glob', type=str, help='trainig data gcs path')
parser.add_argument('--checkpoint_root',  type=str, help='output path to save the checkpoint')
parser.add_argument('--tensorboard_root', type=str, help='output path for tensorboard logs')

args = parser.parse_args()

print(args)

train_model(train_glob=args.train_glob, checkpoint_root=args.checkpoint_root, tensorboard_root=args.tensorboard_root) 

