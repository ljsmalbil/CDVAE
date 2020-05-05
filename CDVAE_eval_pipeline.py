import matplotlib
matplotlib.use('PS')

from networks import p_x_z, p_y_z, q_y_x, q_z_yx
from evaluation import Evaluator, get_y0_y1
from initialisation import init_qz
from preparation import data_preperation
from auxialiary import network, MMDloss, direction, edge_detection, intervention_on_y, var_reduction, adjacency_matrix, determine_threshold, normalise_data, hypothesis_testing, hypothesis_testing_run, hyp_test_main, main_for_var, obtain_distribution
#from mmd_edge import MMDloss, direction, edge_detection
from CDVAE import CDVAE


from cdt.data import AcyclicGraphGenerator
import networkx as nx

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
import os.path

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


def pipeline():

	result_list = []

	# Read the data

	data = pd.read_csv('combined.csv', header = None) 
	true_CM = pd.read_csv('true_CM.csv', header = None)
	true_CM = np.array(true_CM)

	start_time = time.time()

	aupr_matrix, dir_adj_matrix = CDVAE()
	end_time = time.time() - start_time

	print("--- Execution time : %4.4s seconds ---" % end_time)

	shd = SHD(true_CM, dir_adj_matrix, double_for_anticausal=False) 
	print('SHD:', shd)

	aupr, curve = precision_recall(true_CM, aupr_matrix) 
	print('AUPR for aupr_matrix:', aupr)

	#aupr, curve = precision_recall(true_CM, dir_adj_matrix) 
	#print('AUPR for dir_adj_matrix', aupr)

	result_list.append('CDVAE')
	result_list.append(aupr)
	result_list.append(shd)
	result_list.append(end_time)

	# Data
	data = pd.read_csv('combined.csv', header = None)
	data = data.iloc[:,1:]

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

	result_df = pd.DataFrame([result_list, results_pc, results_ccdr, results_ges, results_lingam])

	return result_df


def data():
	mechanism = 'linear' #str(input('Which mechanism do you select to generator the data? '))
	number_parents = 20   #int(input('Desired number of parents? '))
	number_of_data_points = 5000 #int(input('Desired number of data points? '))
	generator = AcyclicGraphGenerator(mechanism, nodes=number_parents, parents_max=3,  noise_coeff=.4, npoints=number_of_data_points)
	data, graph = generator.generate()
	#generator.to_csv('generated_graph')
	# Save true matrix
	true_matrix = retrieve_adjacency_matrix(graph)
	savetxt('true_CM.csv', true_matrix, delimiter=',')

	# Normalise the data
	n = len(data)
	data = np.array(data)
	for i in range(len(data.T)):
	    data[:,i] = (data[:,i]-min(data[:,i]))/(max(data[:,i])-min(data[:,i]))
	data = np.concatenate((np.ones(n).reshape(-1,1), data), axis =1)
	# Save current version in home folder
	savetxt('combined.csv', data, delimiter=',')
	run = 0
	name = 'ArchivedData/combined_' + str(mechanism) + '_' + str(run) + '.csv'  

	# Check if the item alreay exists
	while os.path.exists(name):
		run += 1
		name = 'ArchivedData/combined_' + str(mechanism) + '_' + str(run) + '.csv' 

	run = 0
	name_matrix = 'ArchivedData/True_CM_' + str(mechanism) + '_' + str(run) + '.csv'

	# Check if the item alreay exists
	while os.path.exists(name_matrix):
		run += 1
		name_matrix = 'ArchivedData/True_CM_' + str(mechanism) + '_' + str(run) + '.csv' 

	# Save 
	savetxt(name_matrix, true_matrix, delimiter=',')
	savetxt(name, data, delimiter=',')

result_df = pipeline()
name_of_output_file = 'linear_results_5.csv'  
result_df.to_csv(name_of_output_file)



