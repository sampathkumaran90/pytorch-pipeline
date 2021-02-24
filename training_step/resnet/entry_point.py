import argparse
import os
from os import path
import json
from train import train_model

parser = argparse.ArgumentParser()
parser.add_argument("--train_glob", type=str, help="trainig data path")
parser.add_argument(
    "--model_save_path", type=str, help="output path to save the checkpoint"
)
parser.add_argument(
    "--tensorboard_root", type=str, help="output path to save the tensorboard logs"
)
parser.add_argument(
    "--max_epochs", type=int, default=1, help="Maximum number of epochs"
)
parser.add_argument("--gpus", type=int, default=0, help="Number of gpus for training")
parser.add_argument(
    "--train_batch_size", default=None, type=str, help="Training batch size"
)
parser.add_argument(
    "--val_batch_size", default=None, type=str, help="Validation batch size"
)
parser.add_argument(
    "--train_num_workers", default=4, type=int, help="Training num workers"
)
parser.add_argument(
    "--val_num_workers", default=4, type=int, help="Validation num workers"
)
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument(
    "--accelerator", default=None, type=str, help="Accelerator for multi-gpu training"
)
parser.add_argument('--bucket_name', type=str, help="Bucket name")
parser.add_argument('--folder_name', type=str, help="S3 bucket folder")


args = parser.parse_args()

print(args)

train_model(
    train_glob=args.train_glob,
    model_save_path=args.model_save_path,
    tensorboard_root=args.tensorboard_root,
    max_epochs=args.max_epochs,
    gpus=args.gpus,
    train_batch_size=args.train_batch_size,
    val_batch_size=args.val_batch_size,
    train_num_workers=args.train_num_workers,
    val_num_workers=args.val_num_workers,
    learning_rate=args.learning_rate,
    accelerator=args.accelerator,
    bucket_name=args.bucket_name,
    folder_name=args.folder_name
)
