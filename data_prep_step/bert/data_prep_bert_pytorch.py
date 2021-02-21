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
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
from pathlib import Path

from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice


def run_pipeline(input_options):
    """
    This method downloads the dataset and extract it along with the vocab file

    :param input_options: Input arg parameters
    """
    
    dataset_tar = download_from_url(
        "https://kubeflow-dataset.s3.us-east-2.amazonaws.com/ag_news_csv.tar.gz", root="./")
    extracted_files = extract_archive(dataset_tar)

    ag_news_csv = pv.read_csv("ag_news_csv/train.csv")

    Path(input_options["output"]).mkdir(parents=True, exist_ok=True)
    pq.write_table(ag_news_csv, os.path.join(input_options["output"], "ag_news_data.parquet"))


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
#         "output": "data/test",
#         "VOCAB_FILE": "bert_base_uncased_vocab.txt",
#         "VOCAB_FILE_URL": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt"
#     })
