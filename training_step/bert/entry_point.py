import argparse
import os
from os import path
import json
from news_classifier_bert import train_model

parser = argparse.ArgumentParser()
parser.add_argument('--train_glob', type=str, help='trainig data gcs path')
parser.add_argument('--tensorboard_root', type=str,
                    help='tensorboard path for visualization')
parser.add_argument('--max_epochs', type=int, default=2,
                    help='Maximum number of epochs to run the model')
parser.add_argument('--num_samples', type=int, default=150,
                    help="Maximum number of samples to train on it")
parser.add_argument('--batch_size', type=int, default=16,
                    help="Number of samples per batch")
parser.add_argument('--num_workers', type=int, default=3,
                    help="Number of workers to run(number of cores)")
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help="Learning rate for training")
parser.add_argument('--accelerator', type=str, default="None", help="Acceleration")
parser.add_argument('--model_save_path', type=str, help="Model save directory")

args = parser.parse_args()

train_model(train_glob=args.train_glob, 
            tensorboard_root=args.tensorboard_root,
            max_epochs=args.max_epochs, 
            num_samples=args.num_samples, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            learning_rate=args.learning_rate,
            accelerator=args.accelerator,
            model_save_path=args.model_save_path)
