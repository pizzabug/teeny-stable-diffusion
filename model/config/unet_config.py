from dataclasses import dataclass
from typing import List, Any

from utils.typing import from_str, from_list, from_int, from_bool, from_bool, from_float, from_none, to_float, to_class

@dataclass
class UNetConfig:
    """
        Configuration for UNet2DConditionModel.
        Generated by Quicktype.
    """

    class_name: str
    diffusers_version: str
    act_fn: str
    attention_head_dim: List[int]
    block_out_channels: List[int]
    center_input_sample: bool
    cross_attention_dim: int
    down_block_types: List[str]
    downsample_padding: int
    dual_cross_attention: bool
    flip_sin_to_cos: bool
    freq_shift: int
    in_channels: int
    layers_per_block: int
    mid_block_scale_factor: int
    norm_eps: float
    norm_num_groups: int
    num_class_embeds: None
    only_cross_attention: bool
    out_channels: int
    sample_size: int
    up_block_types: List[str]
    upcast_attention: bool
    use_linear_projection: bool

    @staticmethod
    def from_dict(obj: Any) -> 'UNetConfig':
        assert isinstance(obj, dict)
        class_name = from_str(obj.get("_class_name"))
        diffusers_version = from_str(obj.get("_diffusers_version"))
        act_fn = from_str(obj.get("act_fn"))
        attention_head_dim = from_list(from_int, obj.get("attention_head_dim"))
        block_out_channels = from_list(from_int, obj.get("block_out_channels"))
        center_input_sample = from_bool(obj.get("center_input_sample"))
        cross_attention_dim = from_int(obj.get("cross_attention_dim"))
        down_block_types = from_list(from_str, obj.get("down_block_types"))
        downsample_padding = from_int(obj.get("downsample_padding"))
        dual_cross_attention = from_bool(obj.get("dual_cross_attention"))
        flip_sin_to_cos = from_bool(obj.get("flip_sin_to_cos"))
        freq_shift = from_int(obj.get("freq_shift"))
        in_channels = from_int(obj.get("in_channels"))
        layers_per_block = from_int(obj.get("layers_per_block"))
        mid_block_scale_factor = from_int(obj.get("mid_block_scale_factor"))
        norm_eps = from_float(obj.get("norm_eps"))
        norm_num_groups = from_int(obj.get("norm_num_groups"))
        num_class_embeds = from_int(obj.get("num_class_embeds"))
        only_cross_attention = from_bool(obj.get("only_cross_attention"))
        out_channels = from_int(obj.get("out_channels"))
        sample_size = from_int(obj.get("sample_size"))
        up_block_types = from_list(from_str, obj.get("up_block_types"))
        upcast_attention = from_bool(obj.get("upcast_attention"))
        use_linear_projection = from_bool(obj.get("use_linear_projection"))
        return UNetConfig(class_name, diffusers_version, act_fn, attention_head_dim, block_out_channels, center_input_sample, cross_attention_dim, down_block_types, downsample_padding, dual_cross_attention, flip_sin_to_cos, freq_shift, in_channels, layers_per_block, mid_block_scale_factor, norm_eps, norm_num_groups, num_class_embeds, only_cross_attention, out_channels, sample_size, up_block_types, upcast_attention, use_linear_projection)

    def to_dict(self) -> dict:
        result: dict = {}
        result["_class_name"] = from_str(self.class_name)
        result["_diffusers_version"] = from_str(self.diffusers_version)
        result["act_fn"] = from_str(self.act_fn)
        result["attention_head_dim"] = from_list(
            from_int, self.attention_head_dim)
        result["block_out_channels"] = from_list(
            from_int, self.block_out_channels)
        result["center_input_sample"] = from_bool(self.center_input_sample)
        result["cross_attention_dim"] = from_int(self.cross_attention_dim)
        result["down_block_types"] = from_list(from_str, self.down_block_types)
        result["downsample_padding"] = from_int(self.downsample_padding)
        result["dual_cross_attention"] = from_bool(self.dual_cross_attention)
        result["flip_sin_to_cos"] = from_bool(self.flip_sin_to_cos)
        result["freq_shift"] = from_int(self.freq_shift)
        result["in_channels"] = from_int(self.in_channels)
        result["layers_per_block"] = from_int(self.layers_per_block)
        result["mid_block_scale_factor"] = from_int(
            self.mid_block_scale_factor)
        result["norm_eps"] = to_float(self.norm_eps)
        result["norm_num_groups"] = from_int(self.norm_num_groups)
        result["num_class_embeds"] = from_int(self.num_class_embeds)
        result["only_cross_attention"] = from_bool(self.only_cross_attention)
        result["out_channels"] = from_int(self.out_channels)
        result["sample_size"] = from_int(self.sample_size)
        result["up_block_types"] = from_list(from_str, self.up_block_types)
        result["upcast_attention"] = from_bool(self.upcast_attention)
        result["use_linear_projection"] = from_bool(self.use_linear_projection)
        return result


def welcome_from_dict(s: Any) -> UNetConfig:
    return UNetConfig.from_dict(s)


def welcome_to_dict(x: UNetConfig) -> Any:
    return to_class(UNetConfig, x)