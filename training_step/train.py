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


def train_model(
    train_glob: str,
    checkpoint_root: str,
    tensorboard_root: str
): 
  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

  
    # TODO: this should be in a utils
    def shuffle(old_list):
        size = len(old_list)
        return sample(old_list, size)
  
    # TODO: this should be in a different file for reuse at prediction time
    

    # TODO: this should probably be in a utils file
    class IterableParquetDataset(IterableDataset):
        # TODO: docstring
        # Note: you could use Petastorm (https://github.com/uber/petastorm) to read
        # a parquet file in as a dataloader. I didn't because
        # (1) it's a very heavy, opinionated library
        # (2) it's not 1.0 
        # (3) The long-term maintenance plan wasn't clear
        def __init__(self, dataset, batch_size, process_func=lambda x: x.values(), **kwargs):
            super().__init__()
            self._process_func = process_func
            self._dataset = dataset
            self._row_group_kwargs = kwargs
            self._batch_size = batch_size

        def __iter__(self):
            for piece in shuffle(self._dataset.pieces):
                next_path = piece.path
                filechunk = pq.ParquetFile(self._dataset.fs.open(next_path, 'rb'))
                for row_group_i in shuffle(range(filechunk.num_row_groups)):
                    df = filechunk.read_row_group(row_group_i, **self._row_group_kwargs).to_pandas().sample(frac=1)

                # Create batches from the saved row_sizes
                row_count = len(df)
                boundaries = range(self._batch_size, row_count, self._batch_size)
                boundaries = list(
                    (end - self._batch_size, end)
                    for end in boundaries
                )
                if row_count % self._batch_size != 0:
                    boundaries.append(
                        (row_count - row_count % self._batch_size, row_count)
                    )

                for start, end in boundaries:
                    yield self._process_func(df[start:end])
                    break # TODO: remove! only to test one full epoch

                break
          
    # TODO: This feels like it should be in training_step, and a different one in validation_step
    # So, reduce just to "ToTensor"
    img_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # TODO: this is reused in eval, so it should be in a utils file
    def process_image(rows):
        y = rows['image/class/label'].values
        img_bytes = rows[['image/encoded']].values
        
        img_decompress = np.apply_along_axis(
            lambda img: img_transform(Image.open(BytesIO(img.item()))),
            1,
            img_bytes
        )

        return img_decompress, y
              
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
