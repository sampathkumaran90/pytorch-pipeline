import pytorch_lightning as pl 
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models 

class CIFARCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False 
        num_ftrs = self.model_conv.fc.in_features
        num_classes = 10
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        out = self.model_conv(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model_conv.fc.parameters(), lr=1e-3)
        return optimizer

        