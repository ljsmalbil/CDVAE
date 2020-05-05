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


def data_preperation(data, target):
    target = target 
    #Select the columns
    data_t = data[0]
    data = data.drop([0], axis=1)

    data_y = data[target]
    data = data.drop([target], axis=1)
    data_x = data

    #Reset col_index
    data_x.columns = list(range(0,len(data_x.columns)))

    #Split into train and test
    idxtrain, ite = train_test_split(data_x, test_size=0.1, random_state=1)
    #train_x, train_y 

    #Split train into training and validation set
    itr, iva = train_test_split(idxtrain, test_size=0.3, random_state=1)
    #training, validation

    #Convert indices to lists 
    #itr = list(idxtrain.index.values)  # convert to itr  

    itr = list(itr.index.values)
    iva = list(iva.index.values)
    ite = list(ite.index.values)

    train = (data_x.iloc[itr,:], data_t[itr], data_y[itr])
    valid = (data_x.iloc[iva,:], data_t[iva], data_y[iva])
    test = (data_x.iloc[ite,:], data_t[ite], data_y[ite])

    # Divide train, valid and test
    (xtr, ttr, ytr) = train
    (xva, tva, yva) = valid
    (xte, tte, yte) = test

    # Concatenate val and train into one np.array
    xalltr, talltr, yalltr = np.concatenate([xtr, xva], axis=0), np.concatenate([ttr, tva], axis=0), np.concatenate(
        [ytr, yva], axis=0)

    # zero mean, unit variance for y during training, use ym & ys to correct when using testset
    ym, ys = np.mean(ytr), np.std(ytr)
    ytr, yva = (ytr - ym) / ys, (yva - ym) / ys

    #Convert series and df to numpy arrays
    ttr = np.array(ttr)
    ytr = np.array(ytr)
    xtr = np.array(xtr)

    #Convert to torch tensors
    xtr = torch.tensor(xtr)
    ytr = torch.tensor(ytr)
    ttr = torch.tensor(ttr)

    #Convert train+val concat to torch tensors
    x_train = torch.tensor(xalltr, requires_grad=True)
    y_train = torch.tensor(yalltr, requires_grad=True)
    t_train = torch.tensor(talltr).float()
    y_train = y_train.view(len(y_train),1)
    t_train = t_train.view(len(y_train),1)

    return x_train, y_train, t_train, xtr, ttr, ytr 

