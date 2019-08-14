import torch
import torch.distributions as D
from torch.utils.data import DataLoader

import numpy as np

from itertools import repeat

def fetch_dataset(dataset):
    if dataset == 'MultivariateNormal':
        dimensionality = 64
        loc = torch.zeros(dimensionality).double()
        scale_tril = torch.tensor(np.tril((np.random.rand(dimensionality, dimensionality)-.5)*2)).double()
        dist = D.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        x_sample = dist.sample((512,)).float()
        return x_sample
    elif dataset == 'BivariateNormal':
        dimensionality = 2
        loc = torch.zeros(dimensionality).double()
        scale_tril = torch.tensor(np.tril((np.random.rand(dimensionality, dimensionality)-.5)*2)).double()
        dist = D.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        x_sample = dist.sample((512,)).float()
        return x_sample
    else:
        raise Exception(f"Unknown dataset {dataset}.")

