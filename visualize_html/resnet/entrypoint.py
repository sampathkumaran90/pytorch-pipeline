import argparse
import os
from os import path
import json
import web_app as wa

parser = argparse.ArgumentParser()
parser.add_argument('--board_path', default="", type=str, help='viz data path')
parser.add_argument('--outputs', default="", type=str, help='viz output data path')
parser.add_argument('--metrics_outputs', default="", type=str, help='viz output data path')


args = parser.parse_args()

if __name__ == "__main__":
    wa.show_viz(board_path=args.board_path, outputs=args.outputs, metrics_outputs=args.metrics_outputs)