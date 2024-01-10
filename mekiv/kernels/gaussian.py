import torch

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GaussianKernel:
    """A Gaussian kernel with a single lengthscale."""
    lengthscale: float | torch.Tensor

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """ Evaluate the Gram matrix of the kernel. """
        distances = torch.cdist(X, Y, p=2)
        return torch.exp(-distances**2 / (2*self.lengthscale**2))
    

@dataclass
class ProductGaussianKernel:
    """A product of Gaussian kernels with a lengthscale for each dimension."""
    lengthscales: Tuple[float] | torch.Tensor

    def __post_init__(self):
        self.dim = len(self.lengthscales)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape[1] == Y.shape[1] == len(self.lengthscales)
        componentwise_distances = (X.unsqueeze(1) - Y.unsqueeze(0)) ** 2  # Shape (N_X, N_Y, DIM)
        componentwise_distances /= 2 * self.lengthscales ** 2
        return torch.exp(-componentwise_distances.sum(dim=2))
    
    def sample_from_bochner(self, no_samples: int) -> torch.Tensor:
        """
        Returns samples from the distribution induced by the kernel.
        In each dimension, the distribution is a Gaussian with zero mean and variance 1/lengthscale^2.
        """
        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.dim, device=self.lengthscales.device),
            covariance_matrix=torch.diag(1/self.lengthscales ** 2).to(self.lengthscales.device)
        )
        samples = dist.sample((no_samples,))
        return samples
    
    @staticmethod
    def evaluate_from_samples(X: torch.Tensor, Y: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """
        Given samples from the Bochner distribution, evaluate the kernel.

        The kernel between two points x and x' is just equal to the mean of cos(<w, x-x'>) over all samples w.
        """
        differences = (X.unsqueeze(1) - Y.unsqueeze(0))  # Shape (N_X, N_Y, DIM)
        inner_prods = torch.einsum('xyi,si->xys', differences, samples)
        return torch.mean(torch.cos(inner_prods), dim=2)