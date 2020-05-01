import random

import numpy as np
import scipy
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def get_embedding(model, input, device):
    batch_size = 64
    if input.shape[0] > batch_size:
        embed = []
        i = 0
        while i <= input.shape[0]-1:
            embed.append(
                model(input[i:i+batch_size].to(device), return_feat=True).detach().cpu())
            i += batch_size
        embed = torch.cat(embed)
    else:
        embed = model(input.to(device), return_feat=True).detach().cpu()
    assert embed.shape[0] == input.shape[0]
    return embed.numpy()
