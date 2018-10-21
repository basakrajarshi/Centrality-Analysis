# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 22:35:44 2018

@author: rajar
"""

import networkx as nx
from math import sqrt

def eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None,
                           weight='weight'):
    """Compute the eigenvector centrality for the graph G.
 
    Eigenvector centrality computes the centrality for a node based on the
    centrality of its neighbors. The eigenvector centrality for node `i` is
 
    .. math::
 
        \mathbf{Ax} = \lambda \mathbf{x}
 
    where `A` is the adjacency matrix of the graph G with eigenvalue `\lambda`.
    By virtue of the Perron–Frobenius theorem, there is a unique and positive
    solution if `\lambda` is the largest eigenvalue associated with the
    eigenvector of the adjacency matrix `A` ([2]_).
 
    Parameters
    ----------
    G : graph
      A networkx graph
 
    max_iter : integer, optional
      Maximum number of iterations in power method.
 
    tol : float, optional
      Error tolerance used to check convergence in power method iteration.
 
    nstart : dictionary, optional
      Starting value of eigenvector iteration for each node.
 
    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
 
    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value.
 
       
    Notes
    ------
    The eigenvector calculation is done by the power iteration method and has
    no guarantee of convergence. The iteration will stop after ``max_iter``
    iterations or an error tolerance of ``number_of_nodes(G)*tol`` has been
    reached.
 
    For directed graphs this is "left" eigenvector centrality which corresponds
    to the in-edges in the graph. For out-edges eigenvector centrality
    first reverse the graph with ``G.reverse()``.
 
    
    """

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
            return x
 
    raise nx.NetworkXError("""eigenvector_centrality():
power iteration failed to converge in %d iterations."%(i+1))""")
    
G = nx.read_adjlist('medici_adj_list.txt')
eigenvector_centralities = eigenvector_centrality(G, max_iter=100, tol=1.0e-6, nstart=None,
                           weight=None)    