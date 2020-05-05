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


class Evaluator(object):
    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None):
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1.view(len(mu1),1) - mu0.view(len(mu0),1)

    def rmse_ite(self, ypred1, ypred0):
        #pred_ite = torch.zeros(true_ite.size())
        pred_ite = np.zeros_like(self.mu0)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1 = self.y[idx1].view(len(self.y[idx1]),1) - ypred0[idx1].view(len(ypred0[idx1]),1)
        ite0 = ypred1[idx0].view(len(ypred1[idx0]),1) - self.y[idx0].view(len(self.y[idx0]),1)
        ite1 = ite1.numpy()
        ite1 = ite1.reshape(len(ite1))
        ite0 = ite0.numpy()
        ite0 = ite0.reshape(len(ite0))
        pred_ite[[idx1]] = ite1
        pred_ite[[idx0]] = ite0
        
        return np.sqrt(np.mean((self.true_ite.numpy().reshape(len(self.true_ite)) - pred_ite)**2))

    def abs_ate(self, ypred1, ypred0):
        #return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite)
        return ((ypred1 - ypred0).mean() - self.true_ite.mean()).abs()

    def pehe(self, ypred1, ypred0):
        return np.sqrt(np.mean(np.square((self.mu1 - self.mu0) - (ypred1 - ypred0))))

    def y_errors(self, y0, y1):
        ypred = (1 - self.t) * y0 + self.t * y1
        ypred_cf = self.t * y0 + (1 - self.t) * y1
        return self.y_errors_pcf(ypred, ypred_cf)
    
    def y_errors_pcf(self, ypred, ypred_cf):
        rmse_factual = np.sqrt(np.mean(np.square(ypred - self.y)))
        rmse_cfactual = np.sqrt(np.mean(np.square(ypred_cf - self.y_cf)))
        return rmse_factual, rmse_cfactual

    def calc_stats(self, ypred1, ypred0):
        ite = self.rmse_ite(ypred1, ypred0)
        ate = self.abs_ate(ypred1, ypred0)
        #pehe = self.pehe(ypred1, ypred0)
        return ite, ate   #, pehe
    

def get_y0_y1(p_y_zt_dist, q_y_xt_dist, q_z_tyx_dist, x_train, t_train, L=1):
    y_infer = q_y_x_dist(x_train.float())
    # use inferred y
    xy = torch.cat((x_train.float(), y_infer.mean), 1)  # TODO take mean?
    z_infer = q_z_yx_dist(xy=xy)
    # Manually input zeros and ones
    y0 = p_y_z_dist(z_infer.mean).mean  # TODO take mean?
    y0 = y0.cpu().detach().numpy()
    y0_mean = y0.mean()
    y1 = p_y_z_dist(z_infer.mean).mean  # TODO take mean?

    return y0, y1.cpu().detach().numpy(), y0_mean


