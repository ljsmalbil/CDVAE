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

from networks import p_x_z, p_y_z, q_y_x, q_z_yx
from evaluation import Evaluator, get_y0_y1
from initialisation import init_qz
from preparation import data_preperation
from auxialiary import network, intervention_on_y, var_reduction, adjacency_matrix, determine_threshold, normalise_data, hypothesis_testing, hypothesis_testing_run, hyp_test_main, main_for_var, obtain_distribution


from cdt.metrics import SHD
from numpy.random import randint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class MMDloss(torch.nn.Module):
    """**[torch.nn.Module]** Maximum Mean Discrepancy Metric to compare
    empirical distributions.
    The MMD score is defined by:
    .. math::
        \\widehat{MMD_k}(\\mathcal{D}, \\widehat{\\mathcal{D}}) = 
        \\frac{1}{n^2} \\sum_{i, j = 1}^{n} k(x_i, x_j) + \\frac{1}{n^2}
        \\sum_{i, j = 1}^{n} k(\\hat{x}_i, \\hat{x}_j) - \\frac{2}{n^2} 
        \\sum_{i,j = 1}^n k(x_i, \\hat{x}_j)
    where :math:`\\mathcal{D} \\text{ and } \\widehat{\\mathcal{D}}` represent 
    respectively the observed and empirical distributions, :math:`k` represents
    the RBF kernel and :math:`n` the batch size.
    Args:
        input_size (int): Fixed batch size.
        bandwiths (list): List of bandwiths to take account of. Defaults at
            [0.01, 0.1, 1, 10, 100]
        device (str): PyTorch device on which the computation will be made.
            Defaults at ``cdt.SETTINGS.default_device``.
    Inputs: empirical, observed
        Forward pass: Takes both the true samples and the generated sample in any order 
        and returns the MMD score between the two empirical distributions.
        + **empirical** distribution of shape `(batch_size, features)`: torch.Tensor
          containing the empirical distribution
        + **observed** distribution of shape `(batch_size, features)`: torch.Tensor
          containing the observed distribution.
    Outputs: score
        + **score** of shape `(1)`: Torch.Tensor containing the loss value.
    .. note::
        Ref: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, 
        B., & Smola, A. (2012). A kernel two-sample test.
        Journal of Machine Learning Research, 13(Mar), 723-773.
    Example:
        >>> from cdt.utils.loss import MMDloss
        >>> import torch as th
        >>> x, y = th.randn(100,10), th.randn(100, 10)
        >>> mmd = MMDloss(100)  # 100 is the batch size
        >>> mmd(x, y)
        0.0766
    """

    def __init__(self, input_size, bandwidths=None):
        """Init the model."""
        super(MMDloss, self).__init__()
        if bandwidths is None:
            bandwidths = torch.Tensor([0.01, 0.1, 1, 10, 100])
        else:
            bandwidths = bandwidths
        s = torch.cat([torch.ones([input_size, 1]) / input_size,
                    torch.ones([input_size, 1]) / -input_size], 0)

        self.register_buffer('bandwidths', bandwidths.unsqueeze(0).unsqueeze(0))
        self.register_buffer('S', (s @ s.t()))

    def forward(self, x, y):
        X = torch.cat([x, y], 0)
        # dot product between all combinations of rows in 'X'
        XX = X @ X.t()
        # dot product of rows with themselves
        # Old code : X2 = (X * X).sum(dim=1)
        # X2 = XX.diag().unsqueeze(0)
        X2 = (X * X).sum(dim=1).unsqueeze(0)
        # print(X2.shape)
        # exponent entries of the RBF kernel (without the sigma) for each
        # combination of the rows in 'X'
        exponent = -2*XX + X2.expand_as(XX) + X2.t().expand_as(XX)
        b = exponent.unsqueeze(2).expand(-1,-1, self.bandwidths.shape[2]) * -self.bandwidths
        lossMMD = torch.sum(self.S.unsqueeze(2) * b.exp())
        return lossMMD

def direction(P,Q):
    """
    This function computes the mmd between the marginal distribution of a variable and the post-interventional 
    distribution of that variable given some other variable. The idea being that, by the principle of independence
    of cause and mechanism, if the difference between P and Q is small, than the intervention will not have had
    a big impact on the variable. 
    
    Input: P = p(x)
    Input: Q = p(x|do(y=0))
    
    Return: maximum mean discrepancy score
    """

    mmd = MMDloss(len(P))  # 100 is the batch size
    return mmd(P.float(), Q.float())

# Example
x, y = torch.randn(100,10), torch.randn(100, 10)
mmd = MMDloss(100)  # 100 is the batch size
mmd(x, y)



def edge_detection(child, child_train, parent, block = [], mmd_threshold = 0.1):
    """
    This function determines whether there is an edge between two variables X and Y. 
    
    Arg1: Child distribution
    Arg2: Child train data
    Arg3: Parent to be intervened upon, must be a list
    Arg4: Additional: interventional for the additional
    Arg5: Threshold value
    
    Return: 0 if there is no edge, 1 otherwise
    """
    
    start_time = time.time()
    couter_no = 0
    mmd_val_avg = []
    
    for i in range(5):
        y_normal, data_normal = obtain_distribution(data = child_train.clone(), model = child, intervention_list = block)
        y_interven, data_interven = obtain_distribution(data = child_train.clone(), model = child, intervention_list = parent)
        mmd_value = direction(torch.tensor(y_normal[:4000]), torch.tensor(y_interven[:4000]))
        print('The difference between of the Post-I and Pre-I is distribution is:', mmd_value)
        mmd_val_avg.append(mmd_value)
        
        if mmd_value < float(mmd_threshold):
            #print('No parent')
            couter_no += 1
    print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))    
        
    if couter_no > 3:  #change to 3/4
        return 0, y_interven, np.mean(mmd_val_avg)
    else:
        return 1, y_interven, np.mean(mmd_val_avg)

