import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import StochasticWeightAveraging

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


def train_mnist(config, num_epochs=10, auto_scale_batch_size=None, chk_pt=None):
    grad_clip = config["grad_clip"]
    model = ImageCommandConverter(config)
    early_stop_callback = EarlyStopping(
        monitor='val_loss', min_delta=0.00, patience=20, verbose=True, mode="min")
    callbacks = [early_stop_callback]  # + [StochasticWeightAveraging()]
    trainer = pl.Trainer(callbacks=callbacks,
                         default_root_dir="/Users/renat/Documents/my_rovio/checkpoints/",
                         max_epochs=num_epochs,
                         accumulate_grad_batches=1,
                         gradient_clip_val=grad_clip,
                         auto_scale_batch_size=auto_scale_batch_size,
                         log_every_n_steps=1,
                         resume_from_checkpoint=chk_pt)
    if auto_scale_batch_size is not None:
        trainer.tune(model)
    trainer.fit(model=model)


config = {
    "lr": 1e-3,
    "batch_size": 100,
    "transforms": data_transforms,
    "grad_clip": 0.,
    "lr_sh_factor": 0.1,
    "command_weight": 30,
}

if __name__ == "__main__":
    train_mnist(config, num_epochs=200, auto_scale_batch_size=None,
                chk_pt=None)
