"""
Utility function of decoding hash-type SPhGs to node(=vector) and edge(=matrix) type SPhGs
"""

import numpy as np 

def decodePhct(nodesid,edgesid):
    #nodes
    nodes=[]
    v = nodesid
    #for i in range(24):
    while(v!=0):
        nodes.append(v%10)
        v = v//10
    nodes = nodes[::-1]
    nodes = [v for v in nodes if v>0] # delete 0s

    #edges
    edges=[]
    for v in edgesid:
        edge=[]
        edge.append(v//10000)
        edge.append(v%10000//100)
        edge.append(v%100)
        if edge != [0,0,0]: edges.append(edge)

    #print(edges)
    #print(nodes)
    nnodes = len(nodes)
    #assert nnodes == len(edges)+1

    distmat = np.zeros((nnodes,nnodes),int)
    for edge in edges:
        dist=edge[2]
        if dist==0 and edge[0]!=edge[1]: dist=-1
        distmat[edge[0],edge[1]]=dist
        distmat[edge[1],edge[0]]=dist

    return nodes,distmat
