import torch

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Callable, List
from tqdm import tqdm

import mekiv.utils.misc as utils
from mekiv.kernels.gaussian import ProductGaussianKernel


class XModel(torch.nn.Module):
    def __init__(
        self,
        M: torch.Tensor,
        N: torch.Tensor,
        lambda_N: float,
        K_Z1Z1: torch.Tensor,
        K_Z1Z2: torch.Tensor,
        gamma_MN: torch.Tensor,
        gamma_N: torch.Tensor,
        alpha_sampler: Callable[[int], torch.Tensor],
        true_X: torch.Tensor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device
        self.K_Z1Z1 = K_Z1Z1.to(device)
        self.K_Z1Z2 = K_Z1Z2.to(device)

        lambda_N = torch.tensor([lambda_N], device=device)

        self.initial_X = ((M.clone() + N.clone()) / 2).to(device)

        # Initialise X and lambda_X
        self.X = torch.nn.Parameter(self.initial_X)
        self.log_lambda_X = torch.nn.Parameter(torch.log(lambda_N))

        self.M = M.to(device)
        self.N = N.to(device)

        self.true_X = true_X.to(self.device) if true_X is not None else None

        self.gamma_MN = gamma_MN.to(device)
        self.gamma_N = gamma_N.to(device)
        self.alpha_sampler = alpha_sampler

        self.losses = []
        self.distances = []

    @staticmethod
    def compute_labels(
        alpha_samples: torch.Tensor,  # Shape (C, dim)
        exponent: torch.Tensor,  # Shape (n, dim)
        gamma_numerator: torch.Tensor,  # Shape (n, m)
        gamma_denominator: torch.Tensor,  # Shape (n, m)
        multiplier_numerator: torch.Tensor,  # Shape (n, dim)
    ) -> torch.Tensor:  # Shape (C, m, dim)
        # Shape (C,n); the (a,j)th entry is exp(i * alpha_a . n_j)
        exps = torch.exp(1j * (alpha_samples @ exponent.T).type(torch.complex64))

        # Shape (C, m); the (a,j)th entry is gamma_N(z_j) . exp(i * alpha_a . n_j)
        denominator = exps @ gamma_denominator.type(torch.complex64)

        numerator = torch.einsum(
            "jd,jz,aj -> azd",
            multiplier_numerator.type(torch.complex64),
            gamma_numerator.type(torch.complex64),
            exps,
        )

        return numerator / denominator.unsqueeze(-1)

    def compute_loss(
        self, alpha_samples: torch.Tensor, MN_labels: torch.Tensor
    ) -> torch.Tensor:
        alpha_samples = alpha_samples.to(self.device)
        MN_labels = MN_labels.to(self.device)

        n = self.K_Z1Z1.shape[0]
        gamma_X = torch.linalg.solve(
            self.K_Z1Z1
            + n * torch.exp(self.log_lambda_X) * torch.eye(n, device=self.device),
            self.K_Z1Z2,
        )
        X_labels = self.compute_labels(
            alpha_samples=alpha_samples,
            exponent=self.X,
            gamma_numerator=gamma_X,
            gamma_denominator=gamma_X,
            multiplier_numerator=self.X,
        )

        loss = torch.mean(torch.linalg.norm(MN_labels - X_labels, dim=-1) ** 2)
        return loss

    def fit(
        self,
        no_epochs=1000,
        no_alpha_samples=1000,
        lr=1e-2,
        change_alpha_samples_interval: int = 100,
    ) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        mean_sq_dist = lambda T: torch.norm(self.X - T, dim=-1).square().mean().item()

        for epoch in tqdm(range(no_epochs)):
            if epoch % change_alpha_samples_interval == 0:
                alpha_samples = self.alpha_sampler(no_alpha_samples).to(self.device)
                MN_labels = self.compute_labels(
                    alpha_samples=alpha_samples,
                    exponent=self.N,
                    gamma_numerator=self.gamma_MN,
                    gamma_denominator=self.gamma_N,
                    multiplier_numerator=self.M,
                )

            optimizer.zero_grad()
            loss = self.compute_loss(alpha_samples=alpha_samples, MN_labels=MN_labels)
            loss.backward()
            optimizer.step()

            self.losses.append(loss.item())
            if self.true_X is not None:
                # Record distance of estimates of X to true X
                ms_X = mean_sq_dist(self.true_X)
                ms_M = mean_sq_dist(self.M)
                ms_N = mean_sq_dist(self.N)
                ms_MN = mean_sq_dist((self.M + self.N) / 2)
                self.distances.append(ms_X)

                if epoch % (no_epochs // 100) == 0:
                    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
                    print(
                        f"    Mean square distance to | True X: {ms_X:.4f} | M: {ms_M:.4f} | N: {ms_N:.4f} | MN: {ms_MN:.4f}"
                    )


@dataclass
class MEKIV:
    M: torch.Tensor
    N: torch.Tensor
    Y: torch.Tensor
    Z: torch.Tensor

    lmbda_search_space: Iterable[float]
    xi_search_space: Iterable[float]

    no_epochs: int = 1000
    no_alpha_samples: int = 1000
    lr: float = 1e-2
    change_alpha_samples_interval: int = 100

    real_X: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._is_trained = False

        self.MN = torch.hstack((self.M, self.N))

        if self.real_X is not None:
            first, second = utils.rand_split(
                (self.M, self.N, self.Y, self.Z, self.real_X), p=0.50
            )
            self.M1, self.N1, self.Y1, self.Z1, self.real_X1 = first
            self.M2, self.N2, self.Y2, self.Z2, self.real_X2 = second
        else:
            first, second = utils.rand_split((self.M, self.N, self.Y, self.Z), p=0.5)
            self.M1, self.N1, self.Y1, self.Z1 = first
            self.M2, self.N2, self.Y2, self.Z2 = second
            self.real_X1, self.real_X2 = None, None

        self.MN1 = torch.hstack((self.M1, self.N1))
        self.MN2 = torch.hstack((self.M2, self.N2))

        # Median heuristic for lengthscales, computing from all points
        self.N_lengthscales = utils.heuristic_lengthscales(self.N)
        self.M_lengthscales = utils.heuristic_lengthscales(self.M)
        self.MN_lengthscales = utils.heuristic_lengthscales(self.MN)
        self.Z_lengthscales = utils.heuristic_lengthscales(self.Z)

        self.N_kernel = ProductGaussianKernel(self.N_lengthscales)
        self.MN_kernel = ProductGaussianKernel(self.MN_lengthscales)
        self.Z_kernel = ProductGaussianKernel(self.Z_lengthscales)

        self.fitted_X1: None | torch.Tensor = None

    @property
    def n(self) -> int:
        return self.M1.shape[0]

    @property
    def m(self) -> int:
        return self.M2.shape[0]

    def stage_1_tuning(
        self,
        K_X1X1: torch.FloatTensor,
        K_X2X1: torch.FloatTensor,
        K_X2X2: torch.FloatTensor,
        K_Z1Z1: torch.FloatTensor,
        K_Z1Z2: torch.FloatTensor,
        search_space: Iterable[float],
    ) -> Tuple[float, torch.FloatTensor]:
        n = K_X1X1.shape[0]
        m = K_X2X2.shape[0]

        def get_gamma_Z2(lmbda: float) -> torch.FloatTensor:
            gamma_Z2 = torch.linalg.solve(
                K_Z1Z1 + lmbda * n * torch.eye(n, device=K_Z1Z1.device), K_Z1Z2
            )
            return gamma_Z2  # Shape (n, m)

        def objective(lmbda: float) -> float:
            gamma_Z2 = get_gamma_Z2(lmbda)
            loss = (
                torch.trace(
                    K_X2X2 - 2 * K_X2X1 @ gamma_Z2 + gamma_Z2.T @ K_X1X1 @ gamma_Z2
                )
                / m
            )

            return loss.item()

        lmbda, _, fs = utils.minimize(objective, search_space)
        return lmbda.item(), get_gamma_Z2(lmbda.item())

    def stage_2_tuning(
        self,
        W: torch.FloatTensor,
        K_X1X1: torch.FloatTensor,
        Y1: torch.FloatTensor,
        Y2: torch.FloatTensor,
        search_space: Iterable[float],
    ) -> Tuple[float, torch.FloatTensor]:
        def get_alpha(xi: float) -> torch.FloatTensor:
            alpha = torch.linalg.solve(W @ W.T + self.m * xi * K_X1X1, W @ Y2)
            return alpha

        def objective(xi: float) -> float:
            alpha = get_alpha(xi)
            preds = (alpha.T @ K_X1X1).T
            return torch.mean((Y1 - preds) ** 2).float().item()

        xi, _, fs = utils.minimize(objective, search_space)
        return xi.item(), get_alpha(xi.item())

    def train(self) -> None:
        # Compute kernels
        self.K_N1N1 = self.N_kernel(self.N1, self.N1)
        self.K_N2N1 = self.N_kernel(self.N2, self.N1)
        self.K_N2N2 = self.N_kernel(self.N2, self.N2)

        self.K_MN1MN1 = self.MN_kernel(self.MN1, self.MN1)
        self.K_MN2MN1 = self.MN_kernel(self.MN2, self.MN1)
        self.K_MN2MN2 = self.MN_kernel(self.MN2, self.MN2)

        self.K_Z1Z1 = self.Z_kernel(self.Z1, self.Z1)
        self.K_Z1Z2 = self.Z_kernel(self.Z1, self.Z2)

        # Get lambda_N, lambda_MN
        lambda_N, gamma_N_Z2 = self.stage_1_tuning(
            K_X1X1=self.K_N1N1,
            K_X2X1=self.K_N2N1,
            K_X2X2=self.K_N2N2,
            K_Z1Z1=self.K_Z1Z1,
            K_Z1Z2=self.K_Z1Z2,
            search_space=self.lmbda_search_space,
        )
        lambda_MN, gamma_MN_Z2 = self.stage_1_tuning(
            K_X1X1=self.K_MN1MN1,
            K_X2X1=self.K_MN2MN1,
            K_X2X2=self.K_MN2MN2,
            K_Z1Z1=self.K_Z1Z1,
            K_Z1Z2=self.K_Z1Z2,
            search_space=self.lmbda_search_space,
        )

        self._X1_fitter = XModel(
            M=self.M1,
            N=self.N1,
            lambda_N=lambda_N,
            K_Z1Z1=self.K_Z1Z1,
            K_Z1Z2=self.K_Z1Z2,
            gamma_MN=gamma_MN_Z2,
            gamma_N=gamma_N_Z2,
            alpha_sampler=self.N_kernel.sample_from_bochner,
            true_X=self.real_X1,
        )
        self._X1_fitter.fit(
            no_epochs=self.no_epochs,
            no_alpha_samples=self.no_alpha_samples,
            lr=self.lr,
            change_alpha_samples_interval=self.change_alpha_samples_interval,
        )
        self.fitted_X1 = self._X1_fitter.X.detach().cpu()
        self.lambda_X = torch.exp(self._X1_fitter.log_lambda_X).detach().cpu()

        self.X_kernel = ProductGaussianKernel(
            utils.heuristic_lengthscales(self.fitted_X1)
        )
        K_X1X1 = self.X_kernel(self.fitted_X1, self.fitted_X1)

        W = K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1 + self.n * self.lambda_X * torch.eye(self.n), self.K_Z1Z2
        )

        xi, alpha = self.stage_2_tuning(
            W=W,
            K_X1X1=K_X1X1,
            Y1=self.Y1,
            Y2=self.Y2,
            search_space=self.xi_search_space,
        )
        self._alpha = alpha.detach().cpu()

        self._is_trained = True

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        K_Xxtest = self.X_kernel(self.fitted_X1, x)
        return K_Xxtest.T @ self._alpha

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()

        return self._predict(x)

    def losses_distances(self) -> Tuple[List[float], List[float]]:
        if not self._is_trained:
            self.train()

        return self._X1_fitter.losses, self._X1_fitter.distances
