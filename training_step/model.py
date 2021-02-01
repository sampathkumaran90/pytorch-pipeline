import pytorch_lightning as pl 
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, models 

IMAGENET_CLASS_COUNT = 1001

class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # TODO: torchserve does this processing, but maybe it should get set at training time
        # via the pipeline?
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # TODO: have a step in the pipeline change these
            std=[0.229, 0.224, 0.225]
        )

        backbone = models.mnasnet1_0(pretrained=True)
        num_filters = 1280 # TODO: don't have this hard-coded
        self.feature_extractor = backbone.layers

        # Disable training for the MNasNet layer
        for name, param in self.feature_extractor.named_parameters():
            param.requires_grad = False

        # Note: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        # says that the output should not be normalized. So, no Softmax layer
        self.classifier = nn.Linear(num_filters, IMAGENET_CLASS_COUNT)
        self.params_to_update = [param for _, param in self.classifier.named_parameters()]

    # PyLightning's documentation about how to use the different methods isn't very good
    # https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#inference
    def forward(self, raw):
        # Copied from https://github.com/pytorch/vision/blob/master/torchvision/models/mnasnet.py
        self.feature_extractor.eval()
        with torch.no_grad():
            x = self.normalize(raw)
            x = self.feature_extractor(x)
            x = x.mean([2, 3])

        probs = self.classifier(x)
        return probs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(y.shape[1:]) # TODO: why...?
        x = x.view(x.shape[1:])
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params_to_update, lr=1e-3)
        return optimizer

        