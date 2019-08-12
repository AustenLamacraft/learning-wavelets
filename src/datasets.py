import torch
import torch.distributions as D
from torch.utils.data import DataLoader

import numpy as np

from itertools import repeat

def fetch_dataloader(batch_size, dataset):
    if dataset == 'MultivariateNormal':
        dimensionality = 64
        loc = torch.rand(dimensionality).double()
        scale_tril = torch.tensor(np.tril(np.random.rand(dimensionality, dimensionality)+.1)).double()
        dist = D.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        x_sample = dist.sample((batch_size,)).float()
        x_sample = torch.cat(list(repeat(x_sample, 100)))
    else:
        raise Exception(f"Unknown dataset {dataset}.")

    data_loader = DataLoader(dataset=x_sample, batch_size=batch_size)
    return data_loader