# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 09:19:09 2018

@author: rajar
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import glob
import operator

graph = nx.read_adjlist('medici_adj_list.txt')
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

degree_centralities = {}
shortest_paths = {}
harmonic_centralities = {}
sum_of_geodesics = 0
shortest_routes = {}
betweenness_centralities = {}


#Computing the degree centrality for each family

for j in graph.nodes():
    #print(len(graph.edges(j)))
    degree_centralities[j] = len(graph.edges(j))
#    degree_centralities[family[j]] = len(graph.edges(j))
    
#Computing the harmonic centrality for each family

for k in graph.nodes():
    #print(k)
    #print(nx.single_source_shortest_path_length(graph,k))
    shortest_paths = nx.single_source_shortest_path_length(graph,k)
    sum_of_geodesics = sum(shortest_paths.values())/(total_vertices - 1)
    harmonic_centralities[k] = sum_of_geodesics
    shortest_paths = {}

#shortest_paths = nx.single_source_shortest_path_length(graph,8)
#hc = sum(shortest_paths.values())/(total_vertices - 1)
#print(hc)

#for i in graph.nodes():
#    shortest_routes = nx.shortest_path(graph,source=i)
#    print(i, shortest_routes)  

#for i in graph.nodes():
#    print(i, nx.single_source_shortest_path(graph,i))
#    shortest_routes = nx.single_source_shortest_path(graph,i)
#    # if shortest path goes through node i increase counter by 1
#    for i in shortest_routes:
#        if ()
#shortest_routes = nx.all_pairs_shortest_path(graph, cutoff=None)
#print(shortest_routes)

total_paths = 0
list_of_paths = []
vertex_squared = total_vertices**2
passes_through_i = 0

graph_2 = nx.read_adjlist('medici_edge_list.txt')
for j in graph_2.nodes():
    for k in graph_2.nodes():
        path = nx.shortest_path(graph_2, j, k)
        print(path)
        if (len(path) != 1):
            list_of_paths.append(path)
            total_paths += 1
            
for i in graph.nodes():
    for j in list_of_paths:
        if (i in j):
            passes_through_i += 1
    betweenness_centralities[i] = passes_through_i/((total_paths)
    *(total_vertices**2))
    passes_through_i = 0
    

# For each centrality, sort the pairs in decreasing order of importance
sorted_degrees = sorted(degree_centralities.items(), key=operator.itemgetter(1), reverse=True)    
sorted_harmonics = sorted(harmonic_centralities.items(), key=operator.itemgetter(1))
sorted_betweenness = sorted(betweenness_centralities.items(), key=operator.itemgetter(1), reverse=True)    