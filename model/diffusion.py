import os
from typing import List, Optional
import pytorch_lightning as pl
import torch

from argparse import Namespace
from diffusers import AutoencoderKL, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image

from model.config.diffusion_config import DiffusionConfig

class DiffusionModel(pl.LightningModule):
    """
    Diffusion model class.
    """

    def __init__(
        self, 
        diffusion_config : DiffusionConfig,
        unet_config : dict,
        vae_config : dict,
        scheduler_config : dict,
        override_device : str = None
    ):

        super().__init__()

        self.diffusion_config = diffusion_config
        self.base_path = diffusion_config.pretrained_model_name_or_path

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = PNDMScheduler.from_config(scheduler_config)

        # Load CLIP Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.base_path, subfolder="tokenizer", revision=diffusion_config.revision
        )
        
        # Load CLIP
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.base_path, subfolder="text_encoder", revision=diffusion_config.revision
        )#.to(self.device)

        # Load VAE
        self.vae = AutoencoderKL(**vae_config)#.to(self.device)

        # Load VAE weights if they exist
        if (os.path.isfile(self.base_path + "/vae/diffusion_pytorch_model.bin")):
            self.vae.load_state_dict(torch.load(self.base_path + "/vae/diffusion_pytorch_model.bin"))

        # Load UNet
        self.unet = UNet2DConditionModel(**unet_config)#.to(self.device)

        # Load UNet weights if they exist
        if (os.path.isfile(self.base_path + "/unet/diffusion_pytorch_model.bin")):
            self.unet.load_state_dict(torch.load(self.base_path + "/unet/diffusion_pytorch_model.bin"))

        print(self.device)

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.diffusion_config.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=self.diffusion_config.revision,
        )
        
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        if override_device is not None:
            self._move_to(override_device)
        else:
            self.curr_device = None

    #region Diffusion library functions
    """
    This region contains functions from the diffusers library,
    which is covered by the Apache License 2.0.

    Copyright 2022 The HuggingFace Team. All rights reserved.
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this code except in compliance with the License.
    You may obtain a copy of the License at
    
        http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def _numpy_to_pil(images):
        r"""
        Convert a numpy array to a PIL image.
        Pulled from diffusers.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def _decode_latents_float(self, latents):
        r"""
        Decode latents to images.
        Pulled from diffusers.
        """
        latents = 1 / 0.18215 * latents
        images = self.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        
        return images

    def _decode_latents(self, latents):
        r"""
        Decode latents to images.
        Pulled from diffusers.
        """
        images = self._decode_latents_float(latents)
        images = images.cpu().permute(0, 2, 3, 1).float()
        return images.numpy()

    #endregion
    
    def _move_to (self, device):
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.unet = self.unet.to(device)
        self.curr_device = device
        return self

    def _encode_prompt(self, text : List[str] = None, has_attention : bool = False, batch_size : int = None):
        """
        Convert a text prompt or lack thereof into a latent vector.
        """
        # Generate an empty prompt if none is provided
        if text is None and batch_size is None:
            raise Exception("Must provide either text or batch_size")
        elif text is None:
            text = [""] * batch_size

        # Tokenise text
        text_embeds = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        # Do attention mask
        attention_mask = None
        if has_attention:
            attention_mask = text_embeds.attention_mask
            if self.curr_device is not None:
                attention_mask = attention_mask.to(self.curr_device)
        
        # Convert to device
        if (self.curr_device is not None):
            text_embeds = text_embeds.to(self.device)

        # Encode text
        return self.text_encoder(**text_embeds)

    def forward(self, q, img):
        return self.layer(q)

    def forward_with_text_q (
        self, 
        prompt : list[str],
        negative_prompt : list[str] = None,
        height : int=512,
        width : int=512,
        steps = 1,
        guidance_scale: float = 1):

        # Call parameters
        batch_size = len(prompt)
        do_guidance = True if guidance_scale > 1 else False

        # Encode prompt
        text_embeds = self._encode_prompt(prompt).last_hidden_state
        prompt_embeds = text_embeds

        # If we're doing guidance, this includes the negative prompt
        if (do_guidance):
            # Text encoder
            negative_text_embeds = self._encode_prompt(negative_prompt).last_hidden_state
            prompt_embeds = torch.cat([negative_text_embeds, text_embeds])

        # Convert images to latent space
        latent_shape = (batch_size, 4, height // 8, width // 8)
        latents = torch.randn(latent_shape)
        latents = latents * self.noise_scheduler.init_noise_sigma
        if (self.curr_device is not None):
            latents = latents.to(self.curr_device)

        # Prepare the scheduler
        self.noise_scheduler.set_timesteps(steps)
        timesteps = self.noise_scheduler.timesteps

        for i, t in enumerate(timesteps):
            # Duplicate the latent space for each prompt, due to CFG
            latent_ins = torch.cat([latents] * 2) if do_guidance else latents
            if (self.curr_device is not None):
                latent_ins = latent_ins.to(self.curr_device)
            latent_ins = self.noise_scheduler.scale_model_input(latent_ins, t).to(self.device)

            # Predict the noise residual and compute loss
            noise_pred = self.unet(latent_ins, t, encoder_hidden_states=prompt_embeds).sample

            # Do CFG
            if do_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.noise_scheduler.step(noise_pred.to(self.device), t, latents.to(self.device)).prev_sample

        return latents

    def training_step(self, batch, batch_idx):
        img, q = batch
        img_hat = self.forward_with_text_q(q)
        img_hat = self._decode_latents_float(img_hat)
        loss = torch.nn.functional.mse_loss(img_hat, img)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def train_dataloader(self):
        # return a DataLoader
        pass

    def save_model(self):
        self.pipeline.save_pretrained(self.diffusion_config.pretrained_model_name_or_path)