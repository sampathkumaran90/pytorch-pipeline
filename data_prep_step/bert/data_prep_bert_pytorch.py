import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW
)
import argparse
import os
from tqdm import tqdm
import requests
from torchtext.utils import download_from_url, extract_archive
from torchtext.datasets.text_classification import URLS
import sys
import argparse
import logging


def run_pipeline(input_options):
    """
    This method downloads the dataset and extract it along with the vocab file

    :param input_options: Input arg parameters
    """
    
    dataset_tar = download_from_url(
        "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/ag_news_csv.tar.gz", root=input_options["output"])
    extracted_files = extract_archive(dataset_tar)

    bert_vocab = download_from_url(
        input_options["VOCAB_FILE_URL"], root=input_options["output"])


def PrintOptions(options):
    """
    Logging for debugging
    """
    for a in options.items():
        print(a)


def run_pipeline_component(options):
    """
    Method called from entry point to execute the pipeline    
    """

    print("Running data prep job from container")

    logging.getLogger().setLevel(logging.INFO)

    PrintOptions(options)

    run_pipeline(
        options
    )


# if __name__ == "__main__":
#     run_pipeline_component({
#         "output": "./data",
#         "VOCAB_FILE": "bert_base_uncased_vocab.txt",
#         "VOCAB_FILE_URL": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
#     })
