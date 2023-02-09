import gpytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from sklearn import cluster

from torch.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP

from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

########################################
#
# Original Model
#
########################################
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.layer1 = nn.Linear(1, 40)
        self.layer2 = nn.Linear(40, 40)
        # self.layer3 = nn.Linear(40,1)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        # out = self.layer3(out)
        return out


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(
        #     num_mixtures=4, ard_num_dims=40
        # )
        # self.feature_extractor = feature_extractor

    def forward(self, x):
        # z = self.feature_extractor(x)
        # z_normalized = z - z.min(0)[0]
        # z_normalized = 2 * (z_normalized / z_normalized.max(0)[0]) - 1
        # x_normalized = x - x.min(0)[0]
        # x_normalized = 2 * (x_normalized / x_normalized.max(0)[0]) - 1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


########################################
#
# Newer Model
#
########################################
class FCResNet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        features=40,
        depth=2,
        spectral_normalization=True,
        coeff=0.95,
        n_power_iterations=1,
        dropout_rate=0.01,
        num_outputs=None,
    ):
        super().__init__()
        """
        ResFNN architecture

        Introduced in SNGP: https://arxiv.org/abs/2006.10108
        """
        self.first = nn.Linear(input_dim, features)
        self.residuals = nn.ModuleList(
            [nn.Linear(features, features) for i in range(depth)]
        )
        self.dropout = nn.Dropout(dropout_rate)

        if spectral_normalization:
            self.first = spectral_norm_fc(
                self.first, coeff=coeff, n_power_iterations=n_power_iterations
            )

            for i in range(len(self.residuals)):
                self.residuals[i] = spectral_norm_fc(
                    self.residuals[i],
                    coeff=coeff,
                    n_power_iterations=n_power_iterations,
                )

        self.activation = F.relu
        self.num_outputs = num_outputs
        if num_outputs is not None:
            self.last = nn.Linear(features, num_outputs)

    def forward(self, x):
        x = self.first(x)

        for residual in self.residuals:
            x = x + self.dropout(self.activation(residual(x)))

        if self.num_outputs is not None:
            x = self.last(x).squeeze()

        return x


class ApproximateGPModel(ApproximateGP):
    def __init__(self, inducing_points, initial_lengthscale, likelihood):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.likelihood = likelihood

        kernel = RBFKernel()
        kernel.lengthscale = initial_lengthscale * torch.ones_like(kernel.lengthscale)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

######################################
#
# Utility functions for the new model
#
#####################################
class SpectralNormFC(SpectralNorm):
    def compute_weight(self, module, do_power_iteration: bool) -> torch.Tensor:
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor

        # for logging
        sigma_log = getattr(module, self.name + "_sigma")
        sigma_log.copy_(sigma.detach())

        return weight

    @staticmethod
    def apply(
        module: nn.Module,
        coeff: float,
        name: str,
        n_power_iterations: int,
        dim: int,
        eps: float,
    ) -> "SpectralNormFC":
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    "Cannot register two spectral_norm hooks on "
                    "the same parameter {}".format(name)
                )

        fn = SpectralNormFC(name, n_power_iterations, dim, eps)
        fn.coeff = coeff

        weight = module._parameters[name]
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()

            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)

        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        module.register_buffer(fn.name + "_sigma", torch.ones(1))

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


def spectral_norm_fc(
    module,
    coeff: float,
    n_power_iterations: int = 1,
    name: str = "weight",
    eps: float = 1e-12,
    dim: int = None,
):
    """
    Args:
        module (nn.Module): containing module
        coeff (float, optional): coefficient to normalize to
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        name (str, optional): name of weight parameter
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    Example::

        >>> m = spectral_norm_fc(nn.Linear(20, 40), 2.0)
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNormFC.apply(module, coeff, name, n_power_iterations, dim, eps)
    return module


def initial_values(train_dataset, feature_extractor, n_inducing_points):
    steps = 10
    idx = torch.randperm(len(train_dataset))[:1000].chunk(steps)
    f_X_samples = []

    with torch.no_grad():
        for i in range(steps):
            X_sample, _ = train_dataset.sample_task().sample_data(n_inducing_points, noise=0.1)
            if torch.cuda.is_available():
                X_sample = X_sample.cuda()
                feature_extractor = feature_extractor.cuda()

            f_X_samples.append(feature_extractor(X_sample).cpu())

    f_X_samples = torch.cat(f_X_samples)

    initial_inducing_points = _get_initial_inducing_points(
        f_X_samples.numpy(), n_inducing_points
    )
    initial_lengthscale = _get_initial_lengthscale(f_X_samples)

    return initial_inducing_points, initial_lengthscale


def _get_initial_inducing_points(f_X_sample, n_inducing_points):
    kmeans = cluster.MiniBatchKMeans(
        n_clusters=n_inducing_points, batch_size=n_inducing_points * 10
    )
    kmeans.fit(f_X_sample)
    initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)

    return initial_inducing_points


def _get_initial_lengthscale(f_X_samples):
    if torch.cuda.is_available():
        f_X_samples = f_X_samples.cuda()

    initial_lengthscale = torch.pdist(f_X_samples).mean()

    return initial_lengthscale.cpu()
