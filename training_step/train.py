import logging
      
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models

import gcsfs
import os
from random import sample
import pyarrow.parquet as pq
from io import BytesIO
from torch.utils.data import DataLoader, IterableDataset
from torch.multiprocessing import Queue
from PIL import Image
import numpy as np

from model import SimpleCNN
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from training_dataset import IterableParquetDataset, shuffle, process_image

def train_model(
    train_glob: str,
    checkpoint_root: str,
    tensorboard_root: str
): 
  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
          
    fs = gcsfs.GCSFileSystem(
        token='cloud'
    )
    
    logger.log(logging.INFO, 'Creating model')
    model = SimpleCNN()

    logger.log(logging.INFO, 'Opening files from: '.format(train_glob))
    dataset = pq.ParquetDataset(
        train_glob,
        filesystem=fs
    )
    
    logger.log(logging.INFO, 'Creating dataset')
    train_dataset = IterableParquetDataset(
        dataset,
        32,
        process_func=process_image,
        columns=[
        'image/class/label', # TODO: should these be hard-coded...?
        'image/encoded'
        ]
    )

    logger.log(logging.INFO, 'Creating data loader')
    dataloader = DataLoader(
        train_dataset
    ) 
        
    #tboard = TensorBoardLogger(tensorboard_root)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_root, 'weights.ckpt')
    )

    logger.log(logging.INFO, 'Starting training')
    trainer = pl.Trainer(
    #logger=tboard,
    checkpoint_callback=checkpoint_callback,
    max_epochs=1)

    trainer.fit(model, dataloader)

    return checkpoint_callback.best_model_path
