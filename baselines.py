import matplotlib
matplotlib.use('PS')

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
from cdt.causality.graph import CAM

import networkx as nx
from cdt.causality.graph import CAM
from cdt.causality.graph import GIES

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

# Data
data = pd.read_csv('combined.csv', header = None)
data = data.iloc[:,1:]



def baselines(data):
	# Tests
	start_time = time.time()

	obj = PC()
	output = obj.predict(data)

	adj_mat = nx.adjacency_matrix(output).todense()
	output = clr(adj_mat)

	output[np.isnan(output)] = 0 
	output[np.isposinf(output)] = 1

	predicted = retrieve_adjacency_matrix(output)
	true_matrix = pd.read_csv('true_CM.csv', header = None)
	true_matrix = np.array(true_matrix)

	shd = SHD(np.array(true_matrix), predicted, double_for_anticausal=False) 
	aupr, curve = precision_recall(np.array(true_matrix), output) 

	end_time = (time.time() - start_time)
	print("--- Execution time : %4.4s seconds ---" % end_time)

	results_pc = ['PC', aupr, shd, end_time]
	print(results_pc)

	# Tests
	start_time = time.time()

	obj = GES()
	output = obj.predict(data)

	adj_mat = nx.adjacency_matrix(output).todense()
	output = clr(adj_mat)

	output[np.isnan(output)] = 0 
	output[np.isposinf(output)] = 1

	predicted = retrieve_adjacency_matrix(output)
	true_matrix = pd.read_csv('true_CM.csv', header = None)
	true_matrix = np.array(true_matrix)

	shd = SHD(np.array(true_matrix), predicted, double_for_anticausal=False) 
	aupr, curve = precision_recall(np.array(true_matrix), output) 

	end_time = (time.time() - start_time)
	print("--- Execution time : %4.4s seconds ---" % end_time)


	results_ges = ['GES', aupr, shd, end_time]
	print(results_ges)

	# Tests
	start_time = time.time()

	obj = LiNGAM()
	output = obj.predict(data)

	adj_mat = nx.adjacency_matrix(output).todense()
	output = clr(adj_mat)

	output[np.isnan(output)] = 0 
	output[np.isposinf(output)] = 1

	predicted = retrieve_adjacency_matrix(output)
	true_matrix = pd.read_csv('true_CM.csv', header = None)
	true_matrix = np.array(true_matrix)

	shd = SHD(np.array(true_matrix), predicted, double_for_anticausal=False) 
	aupr, curve = precision_recall(np.array(true_matrix), output) 

	end_time = (time.time() - start_time)
	print("--- Execution time : %4.4s seconds ---" % end_time)


	results_lingam = ['LiNGAM', aupr, shd, end_time]
	print(results_lingam)


	# Tests
	start_time = time.time()

	obj = CCDr()
	output = obj.predict(data)

	adj_mat = nx.adjacency_matrix(output).todense()
	output = clr(adj_mat)

	output[np.isnan(output)] = 0 
	output[np.isposinf(output)] = 1

	predicted = retrieve_adjacency_matrix(output)
	true_matrix = pd.read_csv('true_CM.csv', header = None)
	true_matrix = np.array(true_matrix)

	shd = SHD(np.array(true_matrix), predicted, double_for_anticausal=False) 
	aupr, curve = precision_recall(np.array(true_matrix), output) 

	end_time = (time.time() - start_time)
	print("--- Execution time : %4.4s seconds ---" % end_time)


	results_ccdr = ['CCDR', aupr, shd, end_time]
	print(results_ccdr)

	return results_pc, results_ges, results_lingam, results_ccdr


# Tests
start_time = time.time()

obj = CGNN(nruns=1, train_epochs=500, test_epochs=500)
output = obj.predict(data)

adj_mat = nx.adjacency_matrix(output).todense()
output = clr(adj_mat)

output[np.isnan(output)] = 0 
output[np.isposinf(output)] = 1

predicted = retrieve_adjacency_matrix(output)
true_matrix = pd.read_csv('true_CM.csv', header = None)
true_matrix = np.array(true_matrix)

shd = SHD(np.array(true_matrix), predicted, double_for_anticausal=False) 
aupr, curve = precision_recall(np.array(true_matrix), output) 

end_time = (time.time() - start_time)
print("--- Execution time : %4.4s seconds ---" % end_time)


results_cgnn = ['CGNN', aupr, shd, end_time]
print(results_cgnn)
