import torch
from dataclasses import dataclass
from typing import Iterable, Tuple

import mekiv.utils.misc as utils
from mekiv.kernels.gaussian import ProductGaussianKernel
from mekiv.structures.stage_data import KIVStageData


@dataclass
class KIV:
    data: KIVStageData
    lmbda_search_space: Iterable[float]
    xi_search_space: Iterable[float]

    def __post_init__(self):
        self.X_kernel = ProductGaussianKernel(
            lengthscales=utils.heuristic_lengthscales(self.data.all_X)
        )
        self.Z_kernel = ProductGaussianKernel(
            lengthscales=utils.heuristic_lengthscales(self.data.all_Z)
        )

        # Precompute the kernel matrices
        self.K_X1X1 = self.X_kernel(self.data.stage_1.X, self.data.stage_1.X)
        self.K_X2X1 = self.X_kernel(self.data.stage_2.X, self.data.stage_1.X)
        self.K_X2X2 = self.X_kernel(self.data.stage_2.X, self.data.stage_2.X)

        self.K_Z1Z1 = self.Z_kernel(self.data.stage_1.Z, self.data.stage_1.Z)
        self.K_Z1Z2 = self.Z_kernel(self.data.stage_1.Z, self.data.stage_2.Z)

        self._is_trained: bool = False

    def stage_1_tuning(self, search_space: Iterable[float]) -> float:
        """Get the value of lambda that maximizes the stage 1 objective."""

        def objective(lmbda: float) -> float:
            gamma = torch.linalg.solve(
                self.K_Z1Z1
                + len(self.data.stage_1) * lmbda * torch.eye(len(self.data.stage_1)),
                self.K_Z1Z2,
            )

            loss = torch.trace(
                self.K_X2X2 - 2 * self.K_X2X1 @ gamma + gamma.T @ self.K_X1X1 @ gamma,
            ) / len(self.data.stage_2)

            return loss.item()

        lmbda, _, _ = utils.minimize(objective, search_space)
        return lmbda.item()

    def stage_2_tuning(
        self, lmbda: float | torch.Tensor, search_space: Iterable[float]
    ) -> Tuple[float, torch.Tensor]:
        """Get the value of xi that maximizes the stage 2 objective."""
        W = self.K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1
            + len(self.data.stage_1) * lmbda * torch.eye(len(self.data.stage_1)),
            self.K_Z1Z2,
        )

        def get_alpha(xi: float) -> torch.Tensor:
            return torch.linalg.solve(
                W @ W.T + len(self.data.stage_2) * xi * self.K_X1X1,
                W @ self.data.stage_2.Y,
            )

        def objective(xi: float) -> float:
            alpha = get_alpha(xi)
            preds = self.K_X1X1.T @ alpha  # Shape (no_test, dim)
            distances = torch.norm(preds - self.data.stage_1.Y, dim=-1)
            loss = torch.mean(distances**2).float().item()

            return loss

        xi, _, _ = utils.minimize(objective, search_space)
        return xi.item(), get_alpha(xi.item())

    def train(self) -> None:
        self.lmbda = self.stage_1_tuning(self.lmbda_search_space)
        _, self._alpha = self.stage_2_tuning(self.lmbda, self.xi_search_space)
        self._is_trained = True

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()

        K_Xxtest = self.X_kernel(self.data.stage_1.X, x)
        return K_Xxtest.T @ self._alpha
