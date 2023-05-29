from model.train_model import data_transforms, train_rovio
import uuid

config = {
    "lr": 1e-2,
    "batch_size": 100,
    "transforms": data_transforms,
    "grad_clip": 0.,
    "lr_sh_factor": 0.5,
    "command_weight": 30,
    "pretrained_weights": True,
}

run_name = f"resnet50_cw={config['command_weight']}_lr={config['lr']}_{uuid.uuid4()}"
train_rovio(config, num_epochs=200, run_name=run_name, auto_scale_batch_size=None,
            chk_pt=None)
