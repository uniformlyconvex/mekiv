import torch

from mekiv.kernels.gaussian import GaussianKernel, ProductGaussianKernel


class TestGaussianKernel:
    @staticmethod
    def slow_gaussian_kernel(
        X: torch.Tensor, Y: torch.Tensor, lengthscale: float
    ) -> torch.Tensor:
        """Evaluate the Gram matrix of the kernel."""
        result = torch.zeros((X.shape[0], Y.shape[0]))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                result[i, j] = torch.exp(
                    -torch.sum((x - y) ** 2) / (2 * lengthscale**2)
                )
        return result

    @staticmethod
    def test_gaussian_kernel():
        DIM = 4
        N_X = 100
        N_Y = 200

        lengthscale = torch.rand(1)
        kernel = GaussianKernel(lengthscale=lengthscale)

        X = torch.rand(N_X, DIM)
        Y = torch.rand(N_Y, DIM)

        expected = TestGaussianKernel.slow_gaussian_kernel(X, Y, lengthscale)
        actual = kernel(X, Y)

        assert torch.allclose(expected, actual)


class TestProductGaussianKernel:
    @staticmethod
    def test_componentwise_division():
        DIM = 4
        N_X = 100
        N_Y = 200

        lengthscales = torch.rand(DIM)

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        expected = torch.zeros((N_X, N_Y, DIM))
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                for d in range(DIM):
                    expected[i, j, d] = (x[d] - y[d]) ** 2 / (2 * lengthscales[d] ** 2)

        fast = (X.unsqueeze(1) - Y.unsqueeze(0)) ** 2 / (2 * lengthscales**2)
        assert torch.allclose(expected, fast)

    @staticmethod
    def test_multidim_gaussian():
        DIM = 4
        N_X = 100
        N_Y = 200

        lengthscales = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        kernel = ProductGaussianKernel(lengthscales)

        X = torch.rand((N_X, DIM))
        Y = torch.rand((N_Y, DIM))

        individual_kernels = [
            GaussianKernel(lengthscale)(X[:, d].reshape(-1, 1), Y[:, d].reshape(-1, 1))
            for d, lengthscale in enumerate(lengthscales)
        ]
        expected = torch.ones((N_X, N_Y))
        for individual_kernel in individual_kernels:
            expected *= individual_kernel

        actual = kernel(X, Y)
        assert torch.allclose(expected, actual)
