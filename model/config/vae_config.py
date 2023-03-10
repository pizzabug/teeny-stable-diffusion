from dataclasses import dataclass
from typing import List, Any, TypeVar, Callable, Type, cast

from utils.typing import from_str, from_list, from_int, to_class

@dataclass
class VAEConfig:
    """
        Configuration for AutoencoderKL.
        Generated by Quicktype.
    """

    class_name: str
    diffusers_version: str
    name_or_path: str
    act_fn: str
    block_out_channels: List[int]
    down_block_types: List[str]
    in_channels: int
    latent_channels: int
    layers_per_block: int
    norm_num_groups: int
    out_channels: int
    sample_size: int
    up_block_types: List[str]

    @staticmethod
    def from_dict(obj: Any) -> 'VAEConfig':
        assert isinstance(obj, dict)
        class_name = from_str(obj.get("_class_name"))
        diffusers_version = from_str(obj.get("_diffusers_version"))
        name_or_path = from_str(obj.get("_name_or_path"))
        act_fn = from_str(obj.get("act_fn"))
        block_out_channels = from_list(from_int, obj.get("block_out_channels"))
        down_block_types = from_list(from_str, obj.get("down_block_types"))
        in_channels = from_int(obj.get("in_channels"))
        latent_channels = from_int(obj.get("latent_channels"))
        layers_per_block = from_int(obj.get("layers_per_block"))
        norm_num_groups = from_int(obj.get("norm_num_groups"))
        out_channels = from_int(obj.get("out_channels"))
        sample_size = from_int(obj.get("sample_size"))
        up_block_types = from_list(from_str, obj.get("up_block_types"))
        return VAEConfig(class_name, diffusers_version, name_or_path, act_fn, block_out_channels, down_block_types, in_channels, latent_channels, layers_per_block, norm_num_groups, out_channels, sample_size, up_block_types)

    def to_dict(self) -> dict:
        result: dict = {}
        result["_class_name"] = from_str(self.class_name)
        result["_diffusers_version"] = from_str(self.diffusers_version)
        result["_name_or_path"] = from_str(self.name_or_path)
        result["act_fn"] = from_str(self.act_fn)
        result["block_out_channels"] = from_list(from_int, self.block_out_channels)
        result["down_block_types"] = from_list(from_str, self.down_block_types)
        result["in_channels"] = from_int(self.in_channels)
        result["latent_channels"] = from_int(self.latent_channels)
        result["layers_per_block"] = from_int(self.layers_per_block)
        result["norm_num_groups"] = from_int(self.norm_num_groups)
        result["out_channels"] = from_int(self.out_channels)
        result["sample_size"] = from_int(self.sample_size)
        result["up_block_types"] = from_list(from_str, self.up_block_types)
        return result


def vae_config_from_dict(s: Any) -> VAEConfig:
    return VAEConfig.from_dict(s)


def vae_config_to_dict(x: VAEConfig) -> Any:
    return to_class(VAEConfig, x)
