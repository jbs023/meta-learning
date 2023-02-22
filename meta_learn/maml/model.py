import re
import warnings

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class MetaModule(nn.Module):
    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix="", recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items()
            if isinstance(module, MetaModule)
            else [],
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None

        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names

            else:
                key_escape = re.escape(key)
                key_re = re.compile(r"^{0}\.(.+)".format(key_escape))

                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k)
                    for k in all_names
                    if key_re.match(k) is not None
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                "Module `{0}` has no parameter corresponding to the "
                "submodule named `{1}` in the dictionary `params` "
                "provided as an argument to `forward()`. Using the "
                "default parameters for this submodule. The list of "
                "the parameters in `params`: [{2}].".format(
                    self.__class__.__name__, key, ", ".join(all_names)
                ),
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    "The module must be either a torch module "
                    "(inheriting from `nn.Module`), or a `MetaModule`. "
                    "Got type: `{0}`".format(type(module))
                )
        return input


class MetaConv1d(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        return self._conv_forward(input, params["weight"], bias)


class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        return self._conv_forward(input, params["weight"], bias)


class MetaConv3d(nn.Conv3d, MetaModule):
    __doc__ = nn.Conv3d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        return self._conv_forward(input, params["weight"], bias)


class _MetaBatchNorm(_BatchNorm, MetaModule):
    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        weight = params.get("weight", None)
        bias = params.get("bias", None)

        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            weight,
            bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )


class MetaBatchNorm1d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )


class MetaBatchNorm2d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class MetaBatchNorm3d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D input (got {}D input)".format(input.dim()))


class MetaLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        return F.linear(input, params["weight"], bias)


class MetaBilinear(nn.Bilinear, MetaModule):
    __doc__ = nn.Bilinear.__doc__

    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get("bias", None)
        return F.bilinear(input1, input2, params["weight"], bias)


def conv3x3(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1.0, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class ConvMaml(MetaModule):
    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvMaml, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
        )

        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, "features"))
        features = features.view((features.size(0), -1))
        logits = self.classifier(
            features, params=self.get_subdict(params, "classifier")
        )
        return logits
