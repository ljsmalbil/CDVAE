import matplotlib
matplotlib.use('PS')

from networks import p_x_z, p_y_z, q_y_x, q_z_yx
from evaluation import Evaluator, get_y0_y1
from initialisation import init_qz
from preparation import data_preperation
from auxialiary import network, edge_detection, direction, MMDloss, intervention_on_y, var_reduction, adjacency_matrix, determine_threshold, normalise_data, hypothesis_testing, hypothesis_testing_run, hyp_test_main, main_for_var, obtain_distribution
#from mmd_edge import MMDloss, direction, edge_detection

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


"""
							LOG

	line 144-155 (April 20 2020): target2 and target1 swapped. 


"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = pd.read_csv('combined.csv', header = None) 

data.head()

true_CM = pd.read_csv('true_CM.csv', header = None)
true_CM = np.array(true_CM)

### Algorithm I (Training)

def CDVAE(data = data, true_CM = true_CM):
    # Global Training 
    start_time = time.time()

    for k in range(len(data.T) - 1):
        start_time_loop = time.time()
        target = k + 1
        globals()['x_train_'+str(target), 'q_y_x_dist_child_var_'+str(target)] = main_for_var(data, target, data_preperation, network)
        print(str(target)+' ===============================================================================')
        print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time_loop))

    print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))
    
    
    start_time = time.time()
    causality_threshold = 0.1
    undirected_pair_set = []
    undirected_pair_set_for_aupr = np.zeros(((len(data.T) - 1), (len(data.T) - 1)))

    for i in range(len(data.T) - 1):
        target = i + 1
        print(str(target)+' ===============================================================================')
        parent = 1
        for j in range(len(data.T) - 2):
            if parent == target:
                parent = parent + 1

            print('Assumed child is ' + str(target) + '. Assumed parent is ' + str(parent))
            value, y_interven, mmd_val = edge_detection(globals()['x_train_'+str(target), 'q_y_x_dist_child_var_'+str(target)][1], globals()['x_train_'+str(target), 'q_y_x_dist_child_var_'+str(target)][0], parent = [j], block = [])

            if mmd_val >= causality_threshold:
                undirected_pair_set.append((parent, target))
                print(mmd_val)


            undirected_pair_set_for_aupr[target-1,parent-1] = float(mmd_val)         #  .append((parent, target))
            parent = parent + 1


    print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))

    ud_adj_matrix = np.zeros(((len(data.T) - 1),(len(data.T) - 1)))

    for pair in undirected_pair_set:
        updated_pair = list(pair)
        #print(updated_pair)
        #print(adj_matrix[updated_pair[0]-1,updated_pair[1]-1])
        ud_adj_matrix[updated_pair[0]-1,updated_pair[1]-1] = 1
        
        
    directed_pair_set = []

    for pair in undirected_pair_set: 
        target1 = pair[0]
        parent1 = pair[1]

        p_x = data.iloc[:6000,target1]
        p_x = np.array(p_x).reshape(-1,1)

        value1, y_interven1, mmd_val1 = edge_detection(globals()['x_train_'+str(target1), 'q_y_x_dist_child_var_'+str(target1)][1], globals()['x_train_'+str(target1), 'q_y_x_dist_child_var_'+str(target1)][0], parent = [j], block = [])

        print(target1)
        print(parent1)
        print(mmd_val1)

        target2 = pair[1]
        parent2 = pair[0]

        p_x = data.iloc[:6000,target]
        p_x = np.array(p_x).reshape(-1,1)

        value2, y_interven2, mmd_val2 = edge_detection(globals()['x_train_'+str(target2), 'q_y_x_dist_child_var_'+str(target2)][1], globals()['x_train_'+str(target2), 'q_y_x_dist_child_var_'+str(target2)][0], parent = [j], block = [])

        print(target2)
        print(parent2)
        print(mmd_val2)

        if mmd_val1 > mmd_val2:
            print(str(target2) + ' is the child and ' + str(target1) + ' its parent')
            #directed_pair_set.append([target2, target1])  #swap values
            directed_pair_set.append([target1, target2])
        elif mmd_val1 < mmd_val2:
            print(str(target1) + ' is the child and ' + str(target2) + ' its parent')
            #directed_pair_set.append([target1, target2])  #swap values
            directed_pair_set.append([target2, target1])
        else:
            print('Undecided')

    # Create an estimated directed adjacency matrix with columns representing children and rows the parents. 
    dir_adj_matrix = np.zeros(((len(data.T) - 1),(len(data.T) - 1)))

    for pair in directed_pair_set:
        print(pair)
        #print(dir_adj_matrix[updated_pair[0]-1,updated_pair[1]-1])
        dir_adj_matrix[pair[0]-1,pair[1]-1] = 1
        
    print(undirected_pair_set_for_aupr)
    print(dir_adj_matrix)
    
    return undirected_pair_set_for_aupr, dir_adj_matrix

