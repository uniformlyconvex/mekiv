import numpy as np
import time
import torch

from numpy.typing import ArrayLike
from tqdm import tqdm
from typing import Optional, Tuple, Iterable, Callable


def rand_split(arrays: Iterable[ArrayLike], p: float=0.5, seed: int=42) -> Tuple[Tuple[ArrayLike], Tuple[ArrayLike]]:
    """
    Randomly split the arrays in two groups, with the first group having a proportion of p.
    """
    length = len(arrays[0])
    if not all(len(array) == length for array in arrays):
        raise ValueError("All arrays must have the same length.")
    
    rng = np.random.default_rng(seed=seed)
    indices = np.arange(start=0, stop=length)
    rng.shuffle(indices)

    split_index = int(length * p)
    return (
        tuple(array[indices[:split_index]] for array in arrays),
        tuple(array[indices[split_index:]] for array in arrays)
    )


def interpoint_distances(X1: ArrayLike, X2: Optional[ArrayLike]=None) -> torch.Tensor:
    """
    Returns the interpoint Euclidean distances.
    If X2 is provided, return the distances between points in X1 and points in X2.
    Otherwise, return the distances between points in X1 and points in X1.
    """
    if X2 is None:
        X2 = X1

    return torch.cdist(X1, X2, p=2)


def median_interpoint_distance(X1: ArrayLike, X2: Optional[ArrayLike]=None) -> float:
    """
    Return the median interpoint Euclidean distance.
    """
    distances = interpoint_distances(X1, X2)
    return torch.median(distances).item()


def heuristic_lengthscales(X: ArrayLike) -> torch.Tensor:
    """
    Computes the median interpoint distances for each dimension of X.
    This is commonly used as a heuristic for a product of Gaussian kernels.
    """
    return torch.tensor([
        median_interpoint_distance(X[:,i].reshape(-1,1))
        for i in range(X.shape[1])
    ])


def minimize(func: Callable, test_points: ArrayLike) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Minimizes a function over a set of test points.
    Returns the optimal test point, the optimal function value, and the values of the function at all test points.
    """
    tic = time.time()
    
    if isinstance(test_points, torch.Tensor):
        test_points = test_points.detach().cpu().numpy()

    fs = []

    if test_points.shape == ():
        test_points = np.array([test_points])
    
    for x in tqdm(test_points):
        fs.append(func(x))

    
    opt_idx = np.nanargmin(fs)
    opt_x = test_points[opt_idx]
    opt_f = fs[opt_idx]

    toc = time.time()
    print(f"Minimization took {toc-tic:.2f} seconds.")

    return torch.tensor(opt_x), torch.tensor(opt_f), torch.tensor(fs)


def evaluate_mse(predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Evaluates the MSE between predictions and truth."""
    return (predictions - truth).norm(dim=1).square().mean()


def evaluate_log10_mse(predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Evaluates the log10 MSE between predictions and truth."""
    return torch.log10(evaluate_mse(predictions, truth))