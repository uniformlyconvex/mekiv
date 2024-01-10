import torch

from dataclasses import dataclass
from typing import Tuple

from mekiv.structures.stage_data import TestData
from mekiv.utils.misc import evaluate_log10_mse


@dataclass
class Demand:
    rho: float
    seed: int = 0

    @staticmethod
    def psi(t: torch.Tensor) -> torch.Tensor:
        term_1 = (t - 5).pow(4) / 600
        term_2 = torch.exp(-4 * (t - 5).pow(2))
        term_3 = (t / 10) - 2
        return 2 * (term_1 + term_2 + term_3)

    @staticmethod
    def structural_function(X: torch.Tensor) -> torch.Tensor:
        # X is a tensor of shape (N, 3), where X[:,0] = P, X[:,1] = T, X[:,2] = S
        P = X[:, 0]
        T = X[:, 1]
        S = X[:, 2]

        ans = 100 + (10 + P) * S * Demand.psi(T) - 2 * P
        return ans.reshape(-1, 1)

    def _gen_data(self, no_points: int) -> Tuple[torch.Tensor]:
        torch.manual_seed(self.seed)

        # S ~ Uniform{1,...,7}
        S = torch.randint(1, 8, (no_points, 1))

        # T ~ Uniform[0,10]
        T_dist = torch.distributions.Uniform(0, 10)
        T = T_dist.sample((no_points, 1))

        # (C, V) ~ N(0, I_2)
        CV_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(2), covariance_matrix=torch.eye(2)
        )
        CV = CV_dist.sample((no_points,))
        C, V = CV[:, 0], CV[:, 1]
        C = C.reshape(-1, 1)
        V = V.reshape(-1, 1)

        # e ~ N(rho*V, 1-rho^2)
        e_dist = torch.distributions.Normal(
            loc=torch.zeros(1), scale=torch.sqrt(torch.tensor(1 - self.rho**2))
        )
        e = e_dist.sample((no_points,))
        e += self.rho * V

        # P = 25 + (C+3)*psi(T) + V
        P = 25 + (C + 3) * Demand.psi(T) + V

        # X = (P, T, S), Z = (C, T, S)
        X = torch.hstack((P, T, S))
        Z = torch.hstack((C, T, S))

        # Sanity check
        assert X.shape == Z.shape == (no_points, 3)

        # Y = h(X) + e
        Y = Demand.structural_function(X) + e

        return X, Y, Z

    def generate_MEKIV_data(
        self, no_points: int, merror_type: str, merror_scale: float
    ) -> Tuple[torch.Tensor]:
        X, Y, Z = self._gen_data(no_points)

        if merror_type == "gaussian":
            # Add Gaussian noise in each dimension, with standard deviation merror_scale * std(X)
            X_std = X.std(dim=0)

            assert X_std.shape == (3,)

            err_dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(3),
                covariance_matrix=torch.diag(X_std * merror_scale) ** 2,
            )
            delta_M = err_dist.sample((no_points,))
            delta_M = delta_M.reshape(-1, 3)
            delta_N = err_dist.sample((no_points,))
            delta_N = delta_N.reshape(-1, 3)

            M = X + delta_M
            N = X + delta_N

        else:
            raise NotImplementedError(f"merror_type = {merror_type} not implemented.")

        return X, M, N, Y, Z

    def generate_test_data(self, *args, **kwargs) -> TestData:
        # We always use 2800 points for test data, but we allow more arguments
        # for consistency with other designs.
        P = torch.linspace(10, 25, 20)
        T = torch.linspace(0, 10, 20)
        S = torch.tensor(range(1, 8)).float()

        X = torch.cartesian_prod(P, T, S)
        assert X.shape == (2800, 3)

        truth = Demand.structural_function(X).reshape(-1, 1)
        metric = evaluate_log10_mse

        return TestData(X, truth, metric)
