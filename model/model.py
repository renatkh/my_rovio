import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch.nn.functional import one_hot
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from model.data_set import ImageDataSet, get_datasets


class ImageCommandConverter(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.lr_sh_factor = config["lr_sh_factor"]
        self.command_weight = config["command_weight"]

        # init a pretrained resnet
        backbone = models.resnet50(pretrained=config["pretrained_weights"])
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        num_target_classes = 10  # rovio command number
        self.command_classifier = nn.Sequential(
            nn.Linear(num_filters, num_target_classes),
            # nn.ReLU(),
            nn.Softmax(dim=1),
        )
        self.speed_value = nn.Sequential(
            nn.Linear(num_filters, 1),
            nn.Sigmoid()
        )
        self.cont_value = nn.Sequential(
            nn.Linear(num_filters, 1),
            nn.Sigmoid(),
        )
        self.config = config
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.train_set, self.val_set = get_datasets(
            train_fraction=0.8
        )

    def train_dataloader(self):
        train_dataset = ImageDataSet(
            self.train_set, transform=self.config["transforms"]['train'])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=0)
        return train_loader

    def val_dataloader(self):
        val_dataset = ImageDataSet(
            self.val_set, transform=self.config["transforms"]['validation'])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=0)
        return val_loader

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        command = self.command_classifier(representations)
        speed = self.speed_value(representations)
        cont_value = self.cont_value(representations)
        return torch.column_stack([command, speed, cont_value])

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, t = batch
        command = one_hot(t[:, 0].to(torch.int64), 10)
        speed = t[:, 1]
        cont_value = t[:, 2]
        t = torch.column_stack([command, speed, cont_value]).to(torch.float32)
        t_h = self.forward(x)
        loss_command = F.cross_entropy(t_h[:, :10], t[:, :10])
        loss_speed = F.mse_loss(t_h[:, 10], t[:, 10])
        loss_unit = F.mse_loss(t_h[:, 11], t[:, 11])
        loss = loss_command*self.command_weight + loss_speed + loss_unit
        self.log("train_loss_command", loss_command)
        self.log("train_loss_speed", loss_speed)
        self.log("train_loss_unit", loss_unit)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, t = batch
        command = one_hot(t[:, 0].to(torch.int64), 10)
        speed = t[:, 1]
        cont_value = t[:, 2]
        t = torch.column_stack([command, speed, cont_value]).to(torch.float32)
        t_h = self.forward(x)
        loss_command = F.cross_entropy(t_h[:, :10], t[:, :10])
        loss_speed = F.mse_loss(t_h[:, 10], t[:, 10])
        loss_unit = F.mse_loss(t_h[:, 11], t[:, 11])
        val_loss = loss_command * self.command_weight + loss_speed + loss_unit
        self.log("val_loss", val_loss)
        self.log("val_loss_command", loss_command)
        self.log("val_loss_speed", loss_speed)
        self.log("val_loss_unit", loss_unit)
        self.lr_sch.step(val_loss)
        self.log("lr", self.lr_sch._last_lr[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_sch = ReduceLROnPlateau(
            optimizer, factor=self.lr_sh_factor, patience=5)
        return optimizer
