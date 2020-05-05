import matplotlib
matplotlib.use('PS')

from cdt.data import AcyclicGraphGenerator
import networkx as nx
import matplotlib as plt
from matplotlib import pyplot as plt
import time
import numpy as np
from numpy import savetxt
from cdt.data import load_dataset
import os.path

from cdt.metrics import retrieve_adjacency_matrix

"""
{'linear': LinearMechanism,
                          'polynomial': Polynomial_Mechanism,
                          'sigmoid_add': SigmoidAM_Mechanism,
                          'sigmoid_mix': SigmoidMix_Mechanism,
                          'gp_add': GaussianProcessAdd_Mechanism,
                          'gp_mix': GaussianProcessMix_Mechanism,
                          'nn': NN_Mechanism}[causal_mechanism]
"""





def generator(mechanism = 'linear'):

	number_parents = 20   #int(input('Desired number of parents? '))
	number_of_data_points = 5000  #int(input('Desired number of data points? '))

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
	name_matrix = 'ArchivedData/True_CM_' + str(mechanism) + '_' + str(run) + '.csv' 

	# Check if the item alreay exists
	while os.path.exists(name):
		run += 1
		name = 'ArchivedData/combined_' + str(mechanism) + '_' + str(run) + '.csv' 

	# Save 
	savetxt(name_matrix, true_matrix, delimiter=',')
	savetxt(name, data, delimiter=',')



# Generate Graph

mechanism = str(input('Which mechanism do you select to generator the data? '))
number_parents = 20   #int(input('Desired number of parents? '))
number_of_data_points = int(input('Desired number of data points? '))

generator = AcyclicGraphGenerator(mechanism, nodes=number_parents, parents_max=3,  noise_coeff=.4, npoints=number_of_data_points)
data, graph = generator.generate()
#generator.to_csv('generated_graph')

# Save true matrix
true_matrix = retrieve_adjacency_matrix(graph)
savetxt('true_CM.csv', true_matrix, delimiter=',')



plt.figure(figsize=(40,20))
nx.draw_networkx(graph, font_size=10) # The plot function allows for quick visualization of the graph. 
plt.show()

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




