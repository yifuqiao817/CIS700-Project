#Yifu Qiao, CIS 700
import networkx as nx
import time
import random as rd
import numpy as np
from communities.algorithms import louvain_method
from communities.algorithms import girvan_newman
from communities.algorithms import spectral_clustering

starttime = int(time.time()) #Counting Time
node_upperbound = 10000 #Batch Size of Sampling
def give_time():
    print('Time: ', int(time.time()) - starttime,  ' seconds has passed. ')

def random_sample(G,k): #Sample Dataset
    nodes = list(G.nodes)
    rd.shuffle(nodes)
    new_nodelist = nodes[:10000]
    print(k, 'Node List Created', int(time.time()) - starttime)
    new_edgelist = []
    for edge in G.edges:
        if(edge[0] in new_nodelist and edge[1] in new_nodelist):
            new_edgelist.append(edge)
    print(k, 'Edge List Created', int(time.time()) - starttime)
    newG = nx.Graph()
    newG.add_edges_from(new_edgelist)
    print(k, len(new_edgelist), int(time.time()) - starttime)
    nx.write_edgelist(newG,'D:/Datasets/sampled_network_'+ str(k) +'.edgelist')

def graph_sample(): #Keep only first 10000 nodes
    G = nx.read_edgelist('D:/Datasets/higgs-social_network.edgelist')
    print('Graph Loaded', int(time.time()) - starttime)

    new_edgelist = []
    for edge in G.edges:
        if(int(edge[0]) <= node_upperbound and int(edge[1]) <= node_upperbound):
            new_edgelist.append(edge)
    print(new_edgelist)
    print(len(new_edgelist), int(time.time()) - starttime)
    newG = nx.DiGraph()
    newG.add_edges_from(new_edgelist)
    nx.write_edgelist(newG,'D:/Datasets/sampled_network_directed.edgelist')

def find_cluster(n, c): #Find cluster given the element
    for cluster in c:
        if(n in cluster):
            return cluster
    return []

def cluster_accuracy(output, Y, nodes): #Giving the performance
    acclist = []
    cnum_O = 0
    O = {}
    cnum_Y = 0
    R = {}
    for i in range(len(nodes)):
        n = nodes[i]
        if (n not in O.keys()):
            O[n] = cnum_O
            Ocluster = find_cluster(n, output)
            for mem in Ocluster:
                if(mem != n):
                    O[mem] = cnum_O
            cnum_O += 1
    for j in range(len(nodes)):
        n = nodes[j]
        if (n not in R.keys()):
            R[n] = cnum_Y
            Ycluster = find_cluster(n, Y)
            for mem in Ycluster:
                if (mem != n):
                    R[mem] = cnum_Y
            cnum_Y += 1
    Yanswer = []
    Oanswer = []
    for node in nodes:
        Yanswer.append(R[node])
        Oanswer.append(O[node])
    for ind in range(len(Yanswer)):
        if(Yanswer[ind] == Oanswer[ind]):
            acclist.append(1)
        else:
            acclist.append(0)
    return np.mean(acclist)


def baseline(G): #Basic method for comparing
    nodes = G.nodes
    #edges = G.edges
    clusters = []
    node_visited = []
    node_frontier = []
    for n in nodes:
        new_cluster = []
        if(n in node_visited):
            continue
        neighbors = list(G[n].keys())
        node_visited.append(n)
        new_cluster.append(n)
        for newn in neighbors:
            if(newn not in node_visited):
                node_frontier.append(newn)
        while(len(node_frontier) > 0):
            target = node_frontier[0]
            tneighbors = list(G[target].keys())
            for t in tneighbors:
                if((t not in node_visited) and (t not in node_frontier)):
                    node_frontier.append(t)
            node_frontier.remove(target)
            node_visited.append(target)
            new_cluster.append(target)
        clusters.append(new_cluster)
    give_time()
    return clusters

def nxcluster(G): #NetworkX
    ylusters = nx.clustering(G)
    nxclusters = {}
    for node in ylusters.keys():
        out = str(ylusters[node])
        if (out not in nxclusters.keys()):
            nxclusters[out] = []
        nxclusters[out].append(str(node))
    newcluster = []
    for key in nxclusters.keys():
        newcluster.append(nxclusters[key])
    return newcluster

def louvan(G): #Louvan Method
    nlist = list(G.nodes)
    clist = louvain_method(np.array(nx.adjacency_matrix(G,nodelist=nlist).todense()))[0]
    newcluster = []
    for cset in clist:
        c = []
        slist = list(cset)
        for ind in slist:
            c.append(nlist[ind])
        newcluster.append(c)
    return newcluster


def gn_method(G): #Girvan-Newman Method
    nlist = list(G.nodes)
    clist = girvan_newman(np.array(nx.adjacency_matrix(G, nodelist=nlist).todense()))[0]
    newcluster = []
    for cset in clist:
        c = []
        slist = list(cset)
        for ind in slist:
            c.append(nlist[ind])
        newcluster.append(c)
    return newcluster

def spectral(G): #Spectral Clustering
    nlist = list(G.nodes)
    clist = spectral_clustering(np.array(nx.adjacency_matrix(G, nodelist=nlist).todense()), k=5)
    print(clist)
    newcluster = []
    for cset in clist:
        c = []
        slist = list(cset)
        for ind in slist:
            c.append(nlist[ind])
        newcluster.append(c)
    return newcluster




#G = nx.read_edgelist('D:/Datasets/higgs-social_network.edgelist')
#graph_sample()
print('start')
Gsets = []
for i in range(5):
    #random_sample(G, i)
    Gsets.append(nx.read_edgelist('D:/Datasets/sampled_network_'+ str(i) +'.edgelist'))
graph_counter = 0
for graph in Gsets:
    graph_counter += 1
    print(graph_counter, len(list(graph.edges)))
    Y = baseline(graph)
    out_nx = nxcluster(graph)
    print('Graph '+ str(graph_counter) + ', NetworkX: ', cluster_accuracy(out_nx, Y, list(graph.nodes)))
    give_time()
    out_lv = louvan(graph)
    print('Graph ' + str(graph_counter) + ', Louvan: ', cluster_accuracy(out_lv, Y, list(graph.nodes)))
    give_time()
    out_gn = gn_method(graph)
    print('Graph ' + str(graph_counter) + ', GNewman: ', cluster_accuracy(out_gn, Y, list(graph.nodes)))
    give_time()
    out_sc = spectral(graph)
    print('Graph ' + str(graph_counter) + ', Spectral: ', cluster_accuracy(out_sc, Y, list(graph.nodes)))
    give_time()

exit()