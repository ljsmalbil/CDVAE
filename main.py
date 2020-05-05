import matplotlib
matplotlib.use('PS')

from networks import p_x_z, p_y_z, q_y_x, q_z_yx
from evaluation import Evaluator, get_y0_y1
from initialisation import init_qz
from preparation import data_preperation
from auxialiary import network, MMDloss, direction, edge_detection, intervention_on_y, var_reduction, adjacency_matrix, determine_threshold, normalise_data, hypothesis_testing, hypothesis_testing_run, hyp_test_main, main_for_var, obtain_distribution
#from mmd_edge import MMDloss, direction, edge_detection
from CDVAE import CDVAE

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

from cdt.metrics import retrieve_adjacency_matrix
from cdt.metrics import SHD
from cdt.metrics import precision_recall

from cdt.causality.graph import CCDr
from cdt.causality.graph import CGNN
from cdt.causality.graph import GES
from cdt.causality.graph import LiNGAM
from cdt.causality.graph import PC
from cdt.causality.graph import CAM

from cdt.utils.graph import clr
import networkx as nx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read the data
data = pd.read_csv('combined.csv', header = None) 
true_CM = pd.read_csv('true_CM.csv', header = None)
true_CM = np.array(true_CM)

start_time = time.time()

aupr_matrix, dir_adj_matrix = CDVAE()
end_time = time.time() - start_time

print("--- Execution time : %4.4s seconds ---" % end_time)

#Retrieve SHD and AUPR
shd = SHD(true_CM, dir_adj_matrix, double_for_anticausal=False) 
print('SHD:', shd)

aupr, curve = precision_recall(true_CM, aupr_matrix) 
print('AUPR for aupr_matrix:', aupr)
