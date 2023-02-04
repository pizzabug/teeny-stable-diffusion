import os
import sys
import json
import pytorch_lightning as pl
import torch
import torch.utils.data

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.dirtycollate import dirty_collate
from data.unsplashlite import UnsplashLiteDataset
from model.config.diffusion_config import DiffusionConfig
from model.diffusion import DiffusionModel
from utils.hooks import RegularCheckpoint, train_save_checkpoint

def train_denoiser(device='gpu'):

    # Load JSON and deserialise into DiffusionConfig
    config_json = open("config/config.json", "r").read()
    config_dict = json.loads(config_json)
    diffusion_config = DiffusionConfig.from_dict(config_dict)

    # Load JSON and deserialize into UNetConfig
    unet_config_json = open(diffusion_config.config_unet, "r").read()
    unet_config_dict = json.loads(unet_config_json)

    # Load JSON and deserialize into VAEConfig
    vae_config_json = open(diffusion_config.config_vae, "r").read()
    vae_config_dict = json.loads(vae_config_json)

    # Load JSON and deserialize into SchedulerConfig
    scheduler_config_json = open(diffusion_config.config_scheduler, "r").read()
    scheduler_config_dict = json.loads(scheduler_config_json)

    # Create DiffusionModel
    diffusion_model = DiffusionModel(diffusion_config, unet_config_dict, vae_config_dict, scheduler_config_dict)

    # hparams while i'm working on it
    img_dim = 512

    # data
    dataset = UnsplashLiteDataset(root_dir='/mnt/e/Source/unsplash-lite-corpus-preprocess/db', model=diffusion_model, img_dim=img_dim)
    training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

    train_loader = DataLoader(training_set, batch_size=1, collate_fn=dirty_collate)
    val_loader = DataLoader(validation_set, batch_size=1, collate_fn=dirty_collate)

    # Logger
    denoiser_logger = TensorBoardLogger("tb_logs", name="model")

    denoiser_trainer = pl.Trainer(
        accelerator=device, 
        precision=16, 
        limit_train_batches=0.5, 
        callbacks=[
            RegularCheckpoint(
                model=diffusion_model, 
                period=50, 
                do_q=True,
                do_img=False,
            ),
        ], 
        accumulate_grad_batches=25,
        logger=denoiser_logger)
    while True:
        #try:
            # Load checkpoint if it exists 
            denoiser_trainer.fit(diffusion_model, train_loader, val_loader)
        #except Exception as e:
        #    tb = sys.exc_info()[2]
        #    print(e.with_traceback(tb))
#
train_denoiser()