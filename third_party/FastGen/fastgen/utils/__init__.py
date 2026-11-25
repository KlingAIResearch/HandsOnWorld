# SPDX-FileCopyrightText: Copyright (c) 2021 Facebook, Inc. and its affiliates.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import abc

import torch

from fastgen.utils.registry import locate, _convert_target_to_string as _convert_target_to_string
import fastgen.utils.logging_utils as logger


def expand_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Expands the input tensor `x` to have the same
    number of dimensions as the `target` tensor.

    Pads `x` with singleton dimensions on the end.

    # Example

    ```
    x = torch.ones(5)
    target = torch.ones(5, 10, 30, 1, 10)

    x = expand_like(x, target)
    print(x.shape) # <- [5, 1, 1, 1, 1]
    ```

    Args:
        x (torch.Tensor): The input tensor to expand.
        target (torch.Tensor): The target tensor whose shape length
            will be matched.

    Returns:
        torch.Tensor: The expanded tensor `x` with trailing singleton
            dimensions.
    """
    x = torch.atleast_1d(x)
    while len(x.shape) < len(target.shape):
        x = x[..., None]
    return x


def instantiate(cfg, *args, **kwargs):
    """
    Recursively instantiate objects defined in dictionaries by
    "_target_" and arguments.
    Args:
        cfg: a dict-like object with "_target_" that defines the caller, and
            other keys that define the arguments
    Returns:
        object instantiated by cfg
    """

    if isinstance(cfg, list):
        # Specialize for list, because many classes take
        # list[objects] as arguments, such as ResNet, DatasetMapper
        return [instantiate(x) for x in cfg]

    if isinstance(cfg, abc.Mapping) and "_target_" in cfg:
        # conceptually equivalent to hydra.utils.instantiate(cfg) with _convert_=all,
        # but faster: https://github.com/facebookresearch/hydra/issues/1200
        cfg = {k: instantiate(v) for k, v in cfg.items()}
        cls = cfg.pop("_target_")
        cls = instantiate(cls)

        if isinstance(cls, str):
            cls_name = cls
            cls = locate(cls_name)
            assert cls is not None, cls_name
        else:
            try:
                cls_name = cls.__module__ + "." + cls.__qualname__
            except Exception:
                # target could be anything, so the above could fail
                cls_name = str(cls)
        assert callable(cls), f"_target_ {cls} does not define a callable object"
        try:
            additional_kwargs = {}
            additional_kwargs.update(cfg)
            additional_kwargs.update(kwargs)
            return cls(*args, **additional_kwargs)
        except TypeError:
            logger.error(f"Error when instantiating {cls_name}!")
            raise
    return cfg  # return as-is if don't know what to do


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed,
    but returns a dict that describes the call.

    LazyCall object has to be called with only keyword arguments. Positional
    arguments are not yet supported.

    Examples:
    ::
        from fastgen.utils import instantiate, LazyCall

        layer_cfg = LazyCall(nn.Conv2d)(in_channels=32, out_channels=32)
        layer_cfg.out_channels = 64   # can edit it afterwards
        layer = instantiate(layer_cfg)
    """

    def __init__(self, target):
        if not (callable(target) or isinstance(target, (str, abc.Mapping))):
            raise TypeError(f"target of LazyCall must be a callable or defines a callable! Got {target}")
        self._target = target

    def __call__(self, **kwargs):
        target = self._target
        kwargs["_target_"] = target
        return kwargs
