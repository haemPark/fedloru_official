import os
import math
import json
import typing as tp
from dataclasses import dataclass

import einops
from transformers import AutoConfig, AutoModelForImageClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Devleopor's Note 

The model is based on the ReLoRa offical implementation 
(https://github.com/Guitaricet/relora/blob/main/peft_pretraining/relora.py)

The difference with the official implementation is that Conv2d is used instead of Linear.
In case of Linear, M x N weight matrix is able to be splitted into M x r and r x N matrices. 
But, in case of Conv2d, M x N x H x W weight tensor is not able to be splitted into M x r x H x W and r x N x H x W tensors.
When used with two consecutive Conv2d layers, the receptive field will be changed compared to the original layer.
So, the weight tensor is splitted into M x r x H x W and r x N x 1 x 1 tensors.
"""

@dataclass
class LoRUConfig:
    r: int
    alpha: int
    dropout: float
    target_modules: tp.List[str]
    keep_original_weights: bool
    lora_only: bool = False
    trainable_scaling: bool = False

class LoRUConv2dModel(nn.Module):
    def __init__(
            self, 
            model, 
            r=64, 
            alpha=8, 
            dropout=0, 
            target_modules: tp.Union[tp.List[str], str] = [],
            keep_original_weights=True,
            lora_only=False,
            trainable_scaling=False,
            lora_for_higher_rank=False,
        ):
        if r <= 0:
            raise ValueError("r must be positive.")
        
        super(LoRUConv2dModel, self).__init__()
        self.wrapped_model: nn.Module = model
        self.r = max(r, 1)
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.keep_original_weights = keep_original_weights
        self.lora_only = lora_only
        self.trainable_scaling = trainable_scaling
        self.lora_for_higher_rank = lora_for_higher_rank

        self._config = LoRUConfig(
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            keep_original_weights=keep_original_weights,
            lora_only=lora_only,
            trainable_scaling=trainable_scaling,
        )

        # define forward functoin for LoRUModel: same as the wrapped(original) model
        self.forward = self.wrapped_model.forward

        # build low-rank update model for target modules
        target_module_list = target_modules
        if isinstance(target_module_list, str): # make it list if not
            target_module_list = [target_module_list]

        for module_name, module in self.wrapped_model.named_modules():
            # apply low-rank only to conv2d
            if not isinstance(module, nn.Conv2d):
                continue
            # apply low-rank only to the target modules such as "encoder"
            if not any(target_key in module_name for target_key in target_module_list):
                continue
            
            if self.lora_for_higher_rank:
                if module.in_channels <= self.r or module.out_channels <= self.r:
                    continue

            # W(freeze) + AB
            new_module = LoWRank2DConv(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                r=self.r,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                alpha=self.alpha,
                dropout=self.dropout,
                lora_only=self.lora_only,
                trainable_scaling=self.trainable_scaling
            )

            if self.keep_original_weights:
                new_module.weight = nn.Parameter(
                    module.weight.detach().clone(), requires_grad=False
                )
                if module.bias is not None:
                    new_module.bias = module.bias

            if self.lora_only:
                assert not self.keep_original_weights
                module.weight = None

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def merge(self):
        for module in self.modules():
            if isinstance(module, LoWRank2DConv):
                module.merge()

    def merge_and_reinit(self):
        for module in self.modules():
            if isinstance(module, LoWRank2DConv):
                module.merge_and_reinit()

    def save_pretrained(self, path):
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "relora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)
    
    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "relora_config.json"), "r") as f:
            relora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)
        base_model = AutoModelForImageClassification.from_config(config)

        if "keep_original" in relora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {relora_config['keep_original']}")
            relora_config["lora_only"] = not relora_config.pop("keep_original")
            relora_config["keep_original_weights"] = not relora_config["lora_only"]

        if "trainable_scaling" not in relora_config:
            relora_config["trainable_scaling"] = False

        model = cls(base_model, **relora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model

# Low-Rank Module for 2D Convolutional Tensor
class LoWRank2DConv(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        r: int,
        #
        stride: int=1,
        padding: int=0,
        dilation: int=1,
        groups: int=1,
        bias: bool=True,
        #
        alpha: int=1,
        dropout: float=0.1,
        lora_only: bool=False,
        trainable_scaling: bool=False,
        **kwargs,
    ):
        if r <= 0:
            raise ValueError("r must be positive")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.r = max(r, 1)
        #
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.lora_only = lora_only
        self.output_padding = (0, 0)
        self.padding_mode = "zeros"
        self.alpha = alpha
        self.trainable_scaling = trainable_scaling
        #
        if not lora_only:
            nn.Conv2d.__init__(
                self,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias
            )
            self.weight.requires_grad = False
        else:
            # lora update only : only update A and B and no W
            nn.Module.__init__(self)
            self.weight = None
            self.bias = None

        self.loru_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.r,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.loru_up = nn.Conv2d(
            in_channels=self.r,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.dropout = nn.Dropout(p=dropout)

        if self.trainable_scaling:
            self.scaling = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        else:
            self.scaling = self.alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        if not hasattr(self, "loru_down"):
            # we are in nn.Linear calling reset_parameters
            nn.Conv2d.reset_parameters(self)
            return

        if not self.lora_only:
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        nn.init.kaiming_uniform_(self.loru_down.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.loru_up.weight, a=math.sqrt(5))     

    def _post_loru_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()
        else:
            return self.scaling
        
    @torch.no_grad()
    def merge(self):
        if self.lora_only:
            print("WARNING: Skipping merge, because only lora parameters are used")
            return
        self.weight += (
            einops.einsum(
                self.loru_down.weight,
                self.loru_up.weight,
                "r i h3 w3, o r h1 w1 -> o i h3 w3",
            )
            * self._post_loru_scale()
        )

    @torch.no_grad()
    def merge_and_reinit(self):
        if self.lora_only:
            print("WARNING: Skipping merge and reinit, because only lora parameters are used")
            return
        self.weight += (
            einops.einsum(
                self.loru_down.weight,
                self.loru_up.weight,
                "r i h3 w3, o r h1 w1 -> o i h3 w3",
            )
            * self._post_loru_scale()
        )

        nn.init.kaiming_uniform_(self.loru_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.loru_up.weight)
        if self.trainable_scaling:
            nn.init.zeros_(self.scaling)

    def forward(self, x: torch.Tensor):
        if self.lora_only:
            return self.loru_up(self.dropout(self.loru_down(x))) * self._post_loru_scale()

        result = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.r > 0:
            result += self.loru_up(self.dropout(self.loru_down(x))) * self._post_loru_scale()
        return result
    