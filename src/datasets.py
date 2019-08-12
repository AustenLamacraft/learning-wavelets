import torch
import torch.distributions as D
from torch.utils.data import DataLoader

import numpy as np

def fetch_dataloader(args):
    dimensionality = 64
    n_samples = args.batch_size
    loc = torch.rand(dimensionality).double()
    scale_tril = torch.tensor(np.tril(np.random.rand(dimensionality, dimensionality)+.1)).double()
    dist = D.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    x_sample = dist.sample((n_samples,)).float()
    data_loader = DataLoader(dataset=x_sample, batch_size=args.batch_size)
    return data_loader