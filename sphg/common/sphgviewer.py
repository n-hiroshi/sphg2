"""
Generate images of SPhGs with deceded and undecoded expression.
"""
import os,sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D,DrawingOptions
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from sphg.core.graph import Graph

class sphgviewer():
    def __init__(self):
        pass

    def show(self,nodeSPhG,edgeSPhG,node_label=True,edge_label=True,title='',figsize=(6,4),savename='temp.png'):

        nnodes = np.sum(nodeSPhG!=0)
        nedges = np.sum(edgeSPhG!=0)
        nodeSPhG=nodeSPhG[0:nnodes]
        edgeSPhG=edgeSPhG[0:nedges]
        #print(nodeSPhG)
        #print(edgeSPhG)

        nodes = []
        for inode,nodeid in enumerate(nodeSPhG):
            node = []
            v=nodeid
            while(v!=0):
                node.append(v%10)
                v=v//10
                #print(v)
            node.sort()
            nodes.append(node)

        #print(nodes)
        featsFamilySetList = nodes
        
        adjmat = np.zeros((nnodes,nnodes),int)
        for iedge in range(nedges):
            edgeHash = edgeSPhG[iedge]
            idx0 = int(edgeHash//10000)
            edgeHash1 = edgeHash%10000
            idx1 = int(edgeHash1//100)
            dist = int(edgeHash1%100)
            if dist==0 and idx0 != idx1: dist=-1
            adjmat[idx0,idx1] = dist
            adjmat[idx1,idx0] = dist

        #print(adjmat)
        distmat = adjmat

        self.show_woDecode(distmat,featsFamilySetList,title=title,figsize=figsize,savename=savename)

    def show_woDecode(self,distmat,featsFamilySetList,title='',figsize=(6,4),savename=''):
        plt.figure(figsize=figsize)
        self.unit_plot(distmat,featsFamilySetList,title=title)
        if len(savename) >0:
            plt.savefig(savename)
        else:
            plt.show()

    #def show_multiplot(self,distmat_list,featsFamilySetList_list):
    #    fig=plt.figure(figsize=figsize)
    #
    #    ngraph = len(distmat_list)
    #    assert ngraph == len(featsFamilySetList_list)
    #    
    #
    #    for igraph, ( distmat, featsFamilySetList) in enumerate(zip(distmat_list,featsFamilySetList)):
    #        ax = fig.add_subplot(ngraph,1,igraph)
    #        ax = self.unit_plot(distmat,featsFamilySetList)

    #    plt.show()


    def unit_plot(self,distmat,featsFamilySetList,title=''):
        G = nx.from_numpy_matrix(distmat)
        nodecolor = ['gray' for k in G.nodes]
 
        # define colors
        if len(featsFamilySetList)>0:
            for inode,featsFamilySet in enumerate(featsFamilySetList):
              if len(featsFamilySet)>0:
                if featsFamilySet[0]==1:
                        nodecolor[inode] = 'blue'
                elif featsFamilySet[0]==2:
                        nodecolor[inode] = 'red'
                elif featsFamilySet[0]==6:
                        nodecolor[inode] = 'green'
                elif featsFamilySet[0]==8:
                        nodecolor[inode] = 'black'
                elif featsFamilySet[0]==9:
                        nodecolor[inode] = 'black'
                else:
                        nodecolor[inode] = 'yellow'
 
        def make_node_label(featsList):
            featsList = list(set(featsList)) #delete multiplication e.g. [6,6]
            labelstr = ''
            labellist = ['','D','A','N','P','','R','','','']
            for feats in featsList:
                labelstr+=labellist[feats]
            return labelstr

        def make_node_size(featsList):
            featsList = list(set(featsList)) 
            if featsList == [8]: size = 200
            elif featsList == [9]: size = 200
            else: size = 500
            return size

        def addAromaticLabel(dist):
            if dist == 1 or dist == 31 or dist == 61: return ''
            elif dist < 30: return '%d'%dist
            elif dist < 60: return '%d'%(dist-30)#'%d:'%(dist-30)
            else: return '%d::'%(dist-60)#'%d::'%(dist-60)
        edge_labels=dict([((u,v,),"%s" % addAromaticLabel(d['weight'])) for u,v,d in G.edges(data=True)])

        def addAromaticEdgeColor(dist):
            if dist < 30: return 'black'
            else: return 'green'
        edgecolor=["%s" % addAromaticEdgeColor(d['weight']) for u,v,d in G.edges(data=True)]
        def addAromaticEdgeWidth(dist):
            if dist < 30: return 1
            else: return 10
        #edgewidth=["%d" % addAromaticEdgeWidth(d['weight']) for u,v,d in G.edges(data=True)]
        edgewidth=[addAromaticEdgeWidth(d['weight']) for u,v,d in G.edges(data=True)]
 

        for u,v,d in G.edges(data=True):
            G.edges[(u,v)]['weight']=G.edges[(u,v)]['weight']%30
        
        bbox = dict(boxstyle='round, pad=0.0',ec=(1.0, 1.0, 1.0),fc=(1.0, 1.0, 1.0),alpha=0.3 )


        #node_size = 300#[ d['count']*20 for (n,d) in G.nodes(data=True)]
        nodesize = [make_node_size(featsFamilySetList[k]) for k in G.nodes]
        nodelabels=dict([(k,"%s"%make_node_label(featsFamilySetList[k]).lstrip('[').rstrip(']')) for k in G.nodes])

        pos = nx.kamada_kawai_layout(G)

        nx.draw_networkx_labels(G, pos, font_size=20, font_weight="bold", labels=nodelabels)
        nx.draw_networkx_nodes(G, pos, node_color=nodecolor,alpha=0.5, node_size=nodesize)
        nx.draw_networkx_edge_labels(G,pos,font_size=16,bbox=bbox, alpha=1.0, font_weight="bold",edge_labels=edge_labels)
        #nx.draw_networkx_edges(G, pos, alpha=1, edge_color=edgecolor, width=edgewidth)
        nx.draw_networkx_edges(G, pos, edge_color=edgecolor, width=edgewidth, arrows=True)
        #print("================================")
        if len(title)>0: plt.title(title,fontsize=20)
        plt.axis('off')
        plt.tight_layout()
