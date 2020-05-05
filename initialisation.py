import matplotlib
matplotlib.use('PS')


from cdt.metrics import precision_recall
import numpy as np

from argparse import ArgumentParser
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import log_loss

import torch
from torch.distributions import normal
from torch import optim
import csv

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
import sys

from scipy.stats import ttest_rel
from scipy.stats import kruskal
from scipy.stats import ttest_ind

import time

from cdt.metrics import SHD
from numpy.random import randint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




#Functional version code

def init_qz(qz, pz, y, t, x):

    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    #Change q_z_tyx_dist to qz
    optimizer = optim.Adam(qz.parameters(), lr=0.001)

    for i in range(50):
        batch = np.random.choice(idx, 1)

        #hChange xtr, ytr and ttr to x, y and t resp. 
        x_train, y_train, t_train = x[batch], y[batch], t[batch]

        #Rescale 1D tensors
        y_train = y_train.view(1,1)
        t_train = t_train.view(1,1)

        xy = torch.cat((x_train, y_train), 1)

        #Change to qz
        z_infer = qz(xy=xy.float())

        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz
