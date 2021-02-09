import argparse
import os
from os import path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_root', type=str, help='model checkpoint url')
parser.add_argument('--deployed_model_info_path',  type=str, help='file path to put the deployed url endpoint')

args = parser.parse_args()

print(args)

os.makedirs(args.deployed_model_info_path, exist_ok=True)

with open(os.path.join(args.deployed_model_info_path, "url.txt"), 'w') as outfile:
    outfile.write(args.checkpoint_root)

print("End of deploy component")

