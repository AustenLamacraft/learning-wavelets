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
    elif dataset == 'Sines':
        n_sines = 4
        n_samples = 2**13
        loc = torch.zeros(3).double()
        scale_tril = torch.tensor(np.tril((np.random.rand(3, 3)-.5)*2))
        dist = D.MultivariateNormal(loc=loc, scale_tril=scale_tril)
        def lin_comb_of_sines(x):
            sines_data = dist.sample((n_sines,)).float()
            frequencies = (sines_data[:, 2]+1) * 10
            shifts = torch.zeros(frequencies.shape)  # sines_data[:, 0]
            amplitudes = torch.ones(shifts.shape)  # sines_data[:, 1]
            return torch.sum(amplitudes * torch.sin(shifts + frequencies * x.reshape(-1, 1)), dim=1)
        x_sample = torch.stack([lin_comb_of_sines(torch.linspace(0, 1000, n_samples)) for _ in range(0, 256)])
        return x_sample
    else:
        raise Exception(f"Unknown dataset {dataset}.")

