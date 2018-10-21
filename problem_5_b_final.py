# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 00:37:58 2018

@author: rajar
"""
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np


graph_A = nx.read_adjlist('medici_adj_list.txt')

def harm_cent(graph):
    shortest_paths = {}
    harmonic_centralities = {}
    sum_of_geodesics = 0
    total_vertices = len(graph.nodes())
    for k in graph.nodes():
        shortest_paths = nx.single_source_shortest_path_length(graph,k)
        sum_of_geodesics = sum(shortest_paths.values())/(total_vertices - 1)
        harmonic_centralities[k] = sum_of_geodesics
        shortest_paths = {}
    
    sorted_harmonics = sorted(harmonic_centralities.items(), 
                              key=operator.itemgetter(0))
    return sorted_harmonics

hc_A = harm_cent(graph_A)
hc_B = np.zeros((len(graph_A),2))
data = np.zeros((len(graph_A), 1000))

deg_seq = [1,3,2,3,3,1,4,1,6,1,3,0,3,2,4,3]

graph=nx.configuration_model(deg_seq)

dict_nodes_harm = {}
sum_arr = np.zeros((len(graph),5))
sum_arr[:,0] = range(len(graph))
count = 1

# Generating the random graphs from the degree sequence
while (count <= 1000):
    graph=nx.configuration_model(deg_seq)
    hc = harm_cent(graph)
    print(hc)
    for i in range(len(graph)):
        sum_arr[i,1] += hc[i][1]
        data[i, count-1] = hc[i][1]
    count += 1
    #dict_nodes_harm[]

sum_arr[:,1] = sum_arr[:,1]/1000

for i in range(len(graph_A)):
    hc_B[i,0]=i;
    ind = int(hc_A[i][0])
    hc_B[ind,1] = hc_A[i][1]
    
for i in range(len(data)):
    #print(i)
    sum_arr[i,2] = np.percentile(data[i,:], 25) - hc_B[i,1] # 25th percentile
    sum_arr[i,3] = np.percentile(data[i,:], 75) - hc_B[i,1] # 75th percentile
    sum_arr[i,4] = sum_arr[i,1] - hc_B[i,1] # Mean value
    
# Plotting the graph
fig, ax = plt.subplots()
ax.plot(sum_arr[:,0]+1.,sum_arr[:,2],'--',color='red')
ax.plot(sum_arr[:,0]+1.,sum_arr[:,3],'--',color='red')
ax.fill_between(sum_arr[:,0]+1., sum_arr[:,2], sum_arr[:,3],facecolor='gray', alpha=0.35)
ax.plot(sum_arr[:,0]+1.,sum_arr[:,4],'-o',lw=2,color='blue')
ax.plot(np.linspace(1, 16, 100, endpoint=True),np.zeros((100,1)),'--',lw=1,color='black')
plt.xlabel('vertex label')
plt.ylabel('difference')
plt.savefig('diff_vs_vertexlabel-plot-2.png', dpi = 300)
plt.show