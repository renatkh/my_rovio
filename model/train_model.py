import os
import uuid

import mlflow.pytorch
import pytorch_lightning as pl
from mlflow import MlflowClient
from pytorch_lightning.callbacks import (ModelCheckpoint,
                                         StochasticWeightAveraging)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchvision import transforms

from model.model import ImageCommandConverter

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ]), }


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def train_rovio(config, run_name, num_epochs=10, auto_scale_batch_size=None, chk_pt=None):
    grad_clip = config["grad_clip"]
    model = ImageCommandConverter(config)
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.00, patience=20, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss_command",
        mode="min",
        dirpath="/Users/renat/Documents/my_rovio/checkpoints/",
        filename="rovio-{epoch:02d}-{val_loss:.4f}",
    )
    callbacks = [early_stop_callback, checkpoint_callback] #+ \
       # [StochasticWeightAveraging()]
    trainer = pl.Trainer(callbacks=callbacks,
                         default_root_dir="/Users/renat/Documents/my_rovio/checkpoints/",
                         max_epochs=num_epochs,
                         accumulate_grad_batches=1,
                         gradient_clip_val=grad_clip,
                         auto_scale_batch_size=auto_scale_batch_size,
                         log_every_n_steps=1,
                         resume_from_checkpoint=chk_pt)
    mlflow.pytorch.autolog()
    if auto_scale_batch_size is not None:
        trainer.tune(model)

    experiment_name = "mnist_pretrained"
    # Get the experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Check if the experiment exists
    if experiment is not None:
        experiment_id = experiment.experiment_id
    else:
        # Create the experiment
        experiment_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name,
                          experiment_id=experiment_id,
                          tags={"model": "resnet50",
                                "weights": "mnist_pretrained"},
                          description="parent",) as run:
        for k in config:
            if k not in ['lr']:
                mlflow.log_param(k, config[k])
        trainer.fit(model=model)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    pass
