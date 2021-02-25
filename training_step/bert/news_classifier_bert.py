# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.metrics import Accuracy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets.text_classification import URLS
from torchtext.utils import download_from_url, extract_archive
from transformers import AdamW, BertModel, BertTokenizer
from pytorch_lightning.loggers import TensorBoardLogger
import pyarrow.parquet as pq
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import shutil


class NewsDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length):
        """
        Performs initialization of tokenizer

        :param reviews: AG news text
        :param targets: labels
        :param tokenizer: bert tokenizer
        :param max_length: maximum length of the news text

        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        :return: returns the number of datapoints in the dataframe

        """
        return len(self.reviews)

    def __getitem__(self, item):
        """
        Returns the review text and the targets of the specified item

        :param item: Index of sample review

        :return: Returns the dictionary of review text, input ids, attention mask, targets
        """
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


class BertDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(BertDataModule, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.MAX_LEN = 100
        self.encoding = None
        self.tokenizer = None
        self.args = kwargs

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    @staticmethod
    def process_label(rating):
        rating = int(rating)
        return rating - 1

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        num_samples = self.args["num_samples"]

        data_path = self.args["train_glob"]

        print("\n\nTRAIN GLOB")
        print(data_path)
        print("\n\n")

        df_parquet = pq.ParquetDataset(self.args["train_glob"])

        df = df_parquet.read_pandas().to_pandas()

        df.columns = ["label", "title", "description"]
        df.sample(frac=1)
        df = df.iloc[:num_samples]

        df["label"] = df.label.apply(self.process_label)

        self.tokenizer = BertTokenizer.from_pretrained(self.PRE_TRAINED_MODEL_NAME)

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

        self.df_train, self.df_test = train_test_split(
            df, test_size=0.1, random_state=RANDOM_SEED, stratify=df["label"]
        )
        self.df_val, self.df_test = train_test_split(
            self.df_test,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=self.df_test["label"],
        )

    def create_data_loader(self, df, tokenizer, max_len, batch_size):
        """
        Generic data loader function

        :param df: Input dataframe
        :param tokenizer: bert tokenizer
        :param max_len: Max length of the news datapoint
        :param batch_size: Batch size for training

        :return: Returns the constructed dataloader
        """
        ds = NewsDataset(
            reviews=df.description.to_numpy(),
            targets=df.label.to_numpy(),
            tokenizer=tokenizer,
            max_length=max_len,
        )

        return DataLoader(
            ds, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"]
        )

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.df_train, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.df_val, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.df_test, self.tokenizer, self.MAX_LEN, self.args["batch_size"]
        )
        return self.test_data_loader


class BertNewsClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(BertNewsClassifier, self).__init__()
        self.PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.PRE_TRAINED_MODEL_NAME)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        self.drop = nn.Dropout(p=0.2)
        # assigning labels
        self.class_names = ["World", "Sports", "Business", "Sci/Tech"]
        n_classes = len(self.class_names)

        self.fc1 = nn.Linear(self.bert_model.config.hidden_size, 512)
        self.out = nn.Linear(512, n_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, input_ids, attention_mask):
        """
        :param input_ids: Input data
        :param attention_maks: Attention mask value

        :return: output - Type of news for the given news snippet
        """
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        output = F.relu(self.fc1(output.pooler_output))
        output = self.drop(output)
        output = self.out(output)
        return output

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        input_ids = train_batch["input_ids"].to(self.device)
        attention_mask = train_batch["attention_mask"].to(self.device)
        targets = train_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        self.train_acc(output, targets)
        self.log("train_acc", self.train_acc.compute())
        self.log("train_loss", loss)
        return {"loss": loss, "acc": self.train_acc.compute()}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        input_ids = test_batch["input_ids"].to(self.device)
        attention_mask = test_batch["attention_mask"].to(self.device)
        targets = test_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), targets.cpu())
        self.test_acc(y_hat, targets)
        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": torch.tensor(test_acc)}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """

        input_ids = val_batch["input_ids"].to(self.device)
        attention_mask = val_batch["attention_mask"].to(self.device)
        targets = val_batch["targets"].to(self.device)
        output = self.forward(input_ids, attention_mask)
        loss = F.cross_entropy(output, targets)
        self.val_acc(output, targets)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)
        return {"val_step_loss": loss, "acc": self.val_acc.compute()}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = AdamW(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


def train_model(
    train_glob: str,
    tensorboard_root: str,
    max_epochs: int,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    learning_rate: int,
    accelerator: str,
    model_save_path: str,
    bucket_name: str,
    folder_name: str,
    webapp_path: str
):
    """
    method to train and validate the model

    :param train_glob: Input sentences from the batch
    :param tensorboard_root: Path to save the tensorboard logs
    :param max_epochs: Maximum number of epochs
    :param num_samples: Maximum number of samples to train the model
    :param batch_size: Number of samples ImportError: sys.meta_path is None, Python is likely shutting downper batch
    :param num_workers: Number of cores to train the model
    :param learning_rate: Learning rate used to train the model
    :param accelerator: single or multi GPU
    :param model_save_path: Path for the model to be saved
    :param bucket_name: Name of the S3 bucket
    :param folder_name: Name of the folder to write in S3
    :param webapp_path: Path to save the web content
    """

    if accelerator == "None":
        accelerator = None

    dict_args = {
        "train_glob": train_glob,
        "max_epochs": max_epochs,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": learning_rate,
        "accelerator": accelerator,
    }

    dm = BertDataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BertNewsClassifier(**dict_args)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    if os.path.exists(os.path.join(tensorboard_root, "bert_lightning_kubeflow")):
        shutil.rmtree(os.path.join(tensorboard_root, "bert_lightning_kubeflow"))

    Path(tensorboard_root).mkdir(parents=True, exist_ok=True)

    # Tensorboard root name of the logging directory
    tboard = TensorBoardLogger(tensorboard_root, "bert_lightning_kubeflow")

    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="bert_news_classification_{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        logger=tboard,
        accelerator=accelerator,
        callbacks=[lr_logger, early_stopping],
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
    )
    trainer.fit(model, dm)
    trainer.test()

    s3 = boto3.resource("s3")
    bucket_name = bucket_name
    folder_name = folder_name
    bucket = s3.Bucket(bucket_name)
    s3_path = "s3://" + bucket_name + "/" + folder_name

    for obj in bucket.objects.filter(Prefix=folder_name + "/"):
        s3.Object(bucket.name, obj.key).delete()

    for event_file in os.listdir(
        tensorboard_root + "/bert_lightning_kubeflow/version_0"
    ):
        s3.Bucket(bucket_name).upload_file(
            tensorboard_root + "/bert_lightning_kubeflow/version_0/" + event_file,
            folder_name + "/" + event_file,
            ExtraArgs={"ACL": "public-read"},
        )

    with open("/logdir.txt", "w") as f:
        f.write(s3_path)

    if os.path.exists(webapp_path):
        shutil.rmtree(webapp_path)

    Path(webapp_path).mkdir(parents=True, exist_ok=True)

    rendered_template = """
    <html>
        <head>
            <title>Test Website</title>
        </head>
        <body>
            <div>{}<div>
        </body>
    </html>""".format("Hello World!!!")

    with open(webapp_path + "/test.html", "w") as f:
        f.write(rendered_template)

    s3.Bucket(bucket_name).upload_file(
        webapp_path + "/test.html",
        "test.html",
        ExtraArgs={"ACL": "public-read"},
    )

    with open("/logdirweb.txt", "w") as f:
        f.write("https://kubeflow-dataset.s3.us-east-2.amazonaws.com/test.html")

    # return checkpoint_callback.best_model_path, s3_path


# if __name__ == "__main__":
#     train_model("/home/kumar/Desktop/KUBEFLOW/pytorch-pipeline/data_prep_step/bert/data/test", os.getcwd(),
#                 1, 15, 16, 3, 0.001, None, os.getcwd(), "kubeflow-dataset", "bertViz")
