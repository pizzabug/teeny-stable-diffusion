import os
import pytorch_lightning as pl
import torch

from argparse import Namespace
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from model.config.clip_config import CLIPConfig
from model.config.diffusion_config import DiffusionConfig
from model.config.vae_config import VAEConfig
from model.config.unet_config import UNetConfig

class DiffusionModel(pl.LightningModule):

    def __init__(
        self, 
        diffusion_config : DiffusionConfig,
        clip_config : CLIPConfig,
        unet_config : UNetConfig,
        vae_config : VAEConfig,
    ):

        super().__init__()

        self.diffusion_config = diffusion_config
        self.base_path = diffusion_config.pretrained_model_name_or_path

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(self.base_path, subfolder="scheduler")

        # Load CLIP Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_path, subfolder="tokenizer", revision=diffusion_config.revision
        )
        
        # Load CLIP
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_path, subfolder="text_encoder", revision=diffusion_config.revision
        )

        # Load VAE
        self.vae = AutoencoderKL(
            in_channels=vae_config.in_channels,
            out_channels=vae_config.out_channels,
            down_block_types=vae_config.down_block_types,
            up_block_types=vae_config.up_block_types,
            block_out_channels=vae_config.block_out_channels,
            act_fn=vae_config.act_fn,
            latent_channels=vae_config.latent_channels,
            sample_size=vae_config.sample_size,
        )

        # Load VAE weights if they exist
        if (os.path.isfile(self.base_path + "/vae/diffusion_pytorch_model.bin")):
            self.vae.load_state_dict(torch.load(self.base_path + "/vae/diffusion_pytorch_model.bin"))

        # Load UNet
        self.unet = UNet2DConditionModel(
            sample_size=unet_config.sample_size,
            in_channels=unet_config.in_channels,
            out_channels=unet_config.out_channels,
            center_input_sample=unet_config.center_input_sample,
            flip_sin_to_cos=unet_config.flip_sin_to_cos,
            freq_shift=unet_config.freq_shift,
            down_block_types=unet_config.down_block_types,
            up_block_types=unet_config.up_block_types,
            block_out_channels=unet_config.block_out_channels,
            layers_per_block=unet_config.layers_per_block,
            downsample_padding=unet_config.downsample_padding,
            mid_block_scale_factor=unet_config.mid_block_scale_factor,
            act_fn=unet_config.act_fn,
            norm_num_groups=unet_config.norm_num_groups,
            norm_eps=unet_config.norm_eps,
            cross_attention_dim=unet_config.cross_attention_dim,
            attention_head_dim=unet_config.attention_head_dim,
        )

        # Load UNet weights if they exist
        if (os.path.isdir(self.base_path + "/unet/diffusion_pytorch_model.bin")):
            self.unet.load_state_dict(torch.load(self.base_path + "/unet/diffusion_pytorch_model.bin"))

        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        # return a DataLoader
        pass

    def save_model(self):
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.base_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=self.diffusion_config.revision,
        )
        pipeline.save_pretrained(args.output_dir)