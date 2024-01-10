import torch

from mekiv.designs.demand import Demand
from mekiv.models.mekiv import MEKIV
from mekiv.models.kiv import KIV
from mekiv.structures.stage_data import KIVStageData


SEARCH_SPACE = torch.exp(torch.linspace(-8, 3, 100))
LMBDA_SEARCH_SPACE = SEARCH_SPACE
XI_SEARCH_SPACE = SEARCH_SPACE

design = Demand(rho=0.5)
X, M, N, Y, Z = design.generate_MEKIV_data(
    no_points=1000, merror_type="gaussian", merror_scale=2.0
)

test_data = design.generate_test_data(no_points=2800)

MN_stagedata = KIVStageData.from_all_data((M + N) / 2, Y, Z)
kiv_MN = KIV(
    data=MN_stagedata,
    lmbda_search_space=LMBDA_SEARCH_SPACE,
    xi_search_space=XI_SEARCH_SPACE,
)
kiv_MN_preds = kiv_MN.predict(test_data.X)
kiv_MN_mse = test_data.evaluate_preds(kiv_MN_preds)

mekiv = MEKIV(
    M=M,
    N=N,
    Y=Y,
    Z=Z,
    lmbda_search_space=LMBDA_SEARCH_SPACE,
    xi_search_space=XI_SEARCH_SPACE,
    no_epochs=3000,
    no_alpha_samples=1000,
    change_alpha_samples_interval=100,
    lr=0.01,
    real_X=X,
)
mekiv_preds = mekiv.predict(test_data.X)
mekiv_mse = test_data.evaluate_preds(mekiv_preds)

print(mekiv.lambda_X)
print(kiv_MN.lmbda)

print(f"KIV MN MSE: {kiv_MN_mse}")
print(f"MEKIV MSE: {mekiv_mse}")
