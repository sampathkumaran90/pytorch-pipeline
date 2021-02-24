import argparse
import os
from os import path
import json
import confusion_matrix as cm

parser = argparse.ArgumentParser()
parser.add_argument('--board_path', default="", type=str, help='viz data path')
parser.add_argument('--outputs', default="", type=str, help='viz output data path')


args = parser.parse_args()

if __name__ == "__main__":
    cm.show_viz(board_path=args.board_path, outputs=args.outputs)
    #cm.show_viz(board_path="/home/kumar/Desktop/KUBEFLOW/pytorch-pipeline/training_step/bert/bert_lightning_kubeflow", outputs='test')