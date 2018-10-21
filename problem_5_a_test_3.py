# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:47:09 2018

@author: rajar
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import glob
import operator
from math import sqrt

graph = nx.read_adjlist('medici_adj_list.txt')
graph_2 = nx.read_adjlist('medici_edge_list.txt')
total_edges = len(graph.edges())
total_vertices = len(graph.nodes())
mean_degree = 2*(total_edges)/total_vertices
print("The mean degree is",mean_degree)

nodes = graph.nodes()

family_ids = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
families = ["Acciaiuoli", "Albizzi", "Barbadori", "Bischeri", "Castellani", 
            "Ginori", "Guadagni", "Lamberteschi", "Medici", "Pazzi", 
            "Peruzzi", "Pucci", "Ridolfi", "Salviati", "Strozzi", 
            "Tornabuoni"]

family = dict(zip(family_ids, families))

# Computing the degree centrality for each family

def deg_cent(graph):
    degree_centralities = {} # Dictionary for storing the degree centralities
    for j in graph.nodes(): # Iterate through each node in the graph
        degree_centralities[j] = len(graph.edges(j))
        # Storing the number of edges for each node
    
    sorted_degrees = sorted(degree_centralities.items(), 
                            key=operator.itemgetter(1), reverse=True)
    # Sorting the degree centralities in decreasing order of importance
    return sorted_degrees

#Computing the harmonic centrality for each family

def harm_cent(graph):
    shortest_paths = {} # Dictionary for storing the shortest paths from each vertex
    harmonic_centralities = {} # Dictionary for storing the harmonic centralities 
    sum_of_geodesics = 0
    for k in graph.nodes(): # Iterating through each node 
    
        shortest_paths = nx.single_source_shortest_path_length(graph,k)
        # Fidning the SSSP length from each node to all other nodes
        sum_of_geodesics = sum(shortest_paths.values())/(total_vertices - 1)
        # Summing over all the geodesics for each source vertex
        harmonic_centralities[k] = sum_of_geodesics
        # Storing the harmonic centrality for each node
        shortest_paths = {}
        
    
    sorted_harmonics = sorted(harmonic_centralities.items(), 
                              key=operator.itemgetter(1))
    # Sorting the harmonic centralities in decreasing order of importance
    return sorted_harmonics
        
#Computing the eigenvector centralities

def eig_cent(G, max_iter=100, tol=1.0e-6, nstart=None,
                           weight='weight'):
    
    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
        raise nx.NetworkXException("Not defined for multigraphs.")
 
    if len(G) == 0:
        raise nx.NetworkXException("Empty graph.")
 
    if nstart is None:
 
        # choose starting vector with entries of 1/len(G)
        x = dict([(n,1.0/len(G)) for n in G])
    else:
        x = nstart
 
    # normalize starting vector
    s = 1.0/sum(x.values())
    for k in x:
        x[k] *= s
    nnodes = G.number_of_nodes()
 
    # make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
 
        # do the multiplication y^T = x^T A
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
 
        # normalize vector
        try:
            s = 1.0/sqrt(sum(v**2 for v in x.values()))
 
        # this should never be zero?
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
 
        # check convergence
        err = sum([abs(x[n]-xlast[n]) for n in x])
        if err < nnodes*tol:
            #return x
            sorted_eigenvector = sorted(x.items(), 
                                key=operator.itemgetter(1), reverse=True)
            return sorted_eigenvector

#Computing the betweenness centrality for each family

def bet_cent(graph_2):
    betweenness_centralities = {} # Dictionary for storing the betweenness centralities
    total_paths = 0
    list_of_paths = []
    #vertex_squared = total_vertices**2
    passes_through_i = 0
    
    
    for j in graph_2.nodes():
        for k in graph_2.nodes():
            #path = nx.shortest_path(graph_2, j, k)
            paths = nx.all_shortest_paths(graph_2, j, k)
            # Calculating the shortest path between each pair of nodes
            # and storing it in a list
            #print(paths)
            #if (len(path) != 1):
            for path in paths:
                #print(path)
                list_of_paths.append(path)
                total_paths += 1
                
    for i in graph.nodes(): # Iterating through each node
        for j in list_of_paths: # Iterating through all the shortest paths
            if (i in j): # If a node is included in a path 
                passes_through_i += 1 
                # Increment the counter for nunber of paths passing through
        betweenness_centralities[i] = passes_through_i/((total_paths)
        *(total_vertices**2))
        # Update the betweenness centrality dictionary for each node
        passes_through_i = 0 # Reset the counter for the number of paths
        # passing through
        
    sorted_betweenness = sorted(betweenness_centralities.items(), 
                                key=operator.itemgetter(1), reverse=True)
    # Sorting the betweenness centralities in decreasing order of importance

    return sorted_betweenness 

# Generating the final lists for all the sorted centralities

cd = deg_cent(graph)
ch = harm_cent(graph)
ce = eig_cent(graph)
cb = bet_cent(graph_2)  
 
list_of_deg_cent = []
list_of_harm_cent = []
list_of_eig_cent = []
list_of_bet_cent = []

sz = np.shape(cd)
for i in range(sz[0]):
    temp = cd[i]
    ids = int(temp[0])
    print(families[ids])
    list_of_deg_cent.append((families[ids], np.around(temp[1], decimals = 3)))

sz = np.shape(ch)
for i in range(sz[0]):
    temp = ch[i]
    ids = int(temp[0])
    print(families[ids])
    list_of_harm_cent.append((families[ids], np.around(temp[1], decimals = 3)))
    
sz = np.shape(ce)
for i in range(sz[0]):
    temp = ce[i]
    ids = int(temp[0])
    print(families[ids])
    list_of_eig_cent.append((families[ids], np.around(temp[1], decimals = 3)))
    
sz = np.shape(cb)
for i in range(sz[0]):
    temp = cb[i]
    ids = int(temp[0])
    print(families[ids])
    list_of_bet_cent.append((families[ids], np.around(temp[1], decimals = 4)))
    