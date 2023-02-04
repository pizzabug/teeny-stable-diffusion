import json
import torch

from PIL import Image

from model.diffusion import DiffusionModel
from model.config.diffusion_config import DiffusionConfig
from model.config.unet_config import UNetConfig
from model.config.vae_config import VAEConfig

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
diffusion_model : DiffusionModel = diffusion_model._move_to("cuda")

def test(q):
    # generate some boio
    with torch.no_grad():
        latents = diffusion_model.forward_with_text_q([q], steps=50)
        images = diffusion_model._decode_latents(latents)
        pil_images = DiffusionModel.numpy_to_pil(images)
        return pil_images

if __name__ == "__main__":
    test("bruh")