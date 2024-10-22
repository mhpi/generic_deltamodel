import numpy as np
import torch
import random


def randomseed_config(seed):
    if seed is None:  # args['randomseed'] is None:
        # generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        print("random seed updated!")
    else:
        # randomseed = args['randomseed']
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
