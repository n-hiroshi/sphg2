import pytest
import os,sys
import numpy as np
from rdkit import Chem
import sphg.infra.xmol as xmol
import sphg.infra.miner as miner
import networkx as nx


class GraphEditDistanceSimilarity():
    def __init__(self,datadir='.',ver=''):
        self.datadir=datadir
        self.ver=ver


    def calc(self, nodeSPhGs, edgeSPhGs,isphg=0):
        self.nodeSPhGs = nodeSPhGs
        self.edgeSPhGs = edgeSPhGs
        return self.calcVec(isphg)

    def calcVec(self, isphg):
        #(1) confirm the size of SPhGs
        nsphg = len(self.nodeSPhGs)

        ged_vec = np.zeros(nsphg,float)
        # set the primal SPhG
        G = self.getG(isphg,self.nodeSPhGs[isphg],self.edgeSPhGs[isphg])
        #print("### ged[%d]: %f"%(isphg,ged_vec[isphg]))
        print(self.nodeSPhGs[isphg])
        print(self.edgeSPhGs[isphg])

        for jsphg in range(nsphg):
            print("### jsphg %d"%jsphg)
            # set the isphg-th SPhG
            Gj = self.getG(isphg,self.nodeSPhGs[jsphg],self.edgeSPhGs[jsphg])

            ged_vec[jsphg]=self.ged(G,Gj) #ged=0 and ged_vec[isphg]=1 for idendical graphs of G and Gi
            print("### [%d]-ged[%d]: %f"%(isphg,jsphg,ged_vec[jsphg]))
            print(self.nodeSPhGs[jsphg])
            print(self.edgeSPhGs[jsphg])
        
        return ged_vec

    def calcVecEach(self,isphg,nodeSPhGs,edgeSPhGs):
        self.nodeSPhGs = nodeSPhGs
        self.edgeSPhGs = edgeSPhGs
        ged_vec = self.calcVec(isphg).reshape((1,-1))
        np.savetxt(self.datadir + self.ver + '/' + 'ged_vec_%d.csv'%isphg, ged_vec, delimiter=',', fmt='%f')

    def getG(self,isphg,nodesid,edgesid):
        gedweightdict={'1':0,'2':1,'3':2,'4':3,'6':4,'8':5,'9':5,'12':6,'13':7,'14':8,'23':9,'24':10,'34':11,'123':12,'124':13}
        dummymol = Chem.MolFromSmiles('c1ccccc1')
        xmol0=xmol(dummymol) #'c1ccccc1' has no relation to the SPhGs. Input just avoiding an input format error.
        nodes, distmat = xmol0.decodeSPhG(nodesid,edgesid)
        G = nx.from_numpy_matrix(distmat)
        nodesid1 = nodesid[nodesid>0]
        for inode, weight in enumerate(nodesid1):
            G.nodes[inode]['weight'] = gedweightdict[str(weight)]
        return G

    def ged(self,G1,G2):
        ###########################################################
        node_subst_table = [[0,2,2,2,2,2, 1,1,1,2,2,2,1,1],#0    1=[0]
                            [2,0,2,2,2,2, 1,2,2,1,1,2,1,1],#1    2=[1]
                            [2,2,0,2,2,2, 2,2,1,2,1,2,1,2],#2    3=[2]
                            [2,2,2,0,2,2, 2,2,1,2,1,1,2,1],#3    4=[3]
                            [2,2,2,2,0,2, 2,2,2,2,2,2,2,2],#4    6=[5]
                            [2,2,2,2,2,0, 2,2,2,2,2,2,2,2],#5    9=[6]
                            [1,1,2,2,2,2, 0,2,2,2,2,2,2,2],#6   12=[0,1]
                            [1,2,1,2,2,2, 2,0,2,2,2,2,2,2],#7   13=[0,2]
                            [1,2,2,1,2,2, 2,2,0,2,2,2,2,2],#8   14=[0,3]
                            [2,1,1,2,2,2, 2,2,2,0,2,2,2,2],#9   23=[1,2]
                            [2,1,2,1,2,2, 2,2,2,2,0,2,2,2],#10  24=[1,3]
                            [2,2,1,1,2,2, 2,2,2,2,2,0,2,2],#11  34=[2,3]
                            [1,1,1,2,2,2, 2,2,2,2,2,2,0,2],#12 123=[0,1,2]
                            [1,1,2,1,2,2, 2,2,2,2,2,2,2,0]]#13 124=[0,1,3]
 
 
        def node_subst_cost(node1, node2):
            return node_subst_table[node1['weight']][node2['weight']]
 
 
        def node_del_cost(node):
            if node['weight'] ==9:return 0.5 # junction node 9 = Serratosa's Carbon Link Node
            else: return 1.0
 
        def node_ins_cost(node):
            if node['weight'] ==9:return 0.5 # junction node 9 = Serratosa's Carbon Link Node
            else: return 1.0
 
        edgeLengthCostVec =[0.0] + [np.sum(1/np.arange(1,i+1)) for i in range(1,31)]
        #print('edgeLengthCostVec')
        #print(edgeLengthCostVec)
        def edge_subst_cost(edge1, edge2):
            w1=edge1['weight']
            w2=edge2['weight']
            if   w1<=30 and w2<=30:
                return np.abs(edgeLengthCostVec[w1]-edgeLengthCostVec[w2])
            elif w1>30  and w2<=30:
                w1=w1%30
                return np.abs(edgeLengthCostVec[w1]-edgeLengthCostVec[w2])+3.0*w1
            elif w1<=30 and w2>30 :
                w2=w2%30
                return np.abs(edgeLengthCostVec[w1]-edgeLengthCostVec[w2])+3.0*w2
            else:
                w1=w1%30
                w2=w2%30
                return np.abs(edgeLengthCostVec[w1]-edgeLengthCostVec[w2])*10.0
 
 
        def edge_del_cost(edge):
            if edge['weight']<=30:
               return edge['weight']*0.1
            else:
               return edge['weight']%30*1.0
 
        def edge_ins_cost(edge):
            if edge['weight']<=30:
               return edge['weight']*0.1
            else:
               return (edge['weight']%30)*1.0
        ##########################################################
 
        ged=nx.graph_edit_distance(
        G1,
        G2,
        node_subst_cost=node_subst_cost,
        node_del_cost=node_del_cost,
        node_ins_cost=node_ins_cost,
        edge_subst_cost=edge_subst_cost,
        edge_del_cost=edge_del_cost,
        edge_ins_cost=edge_ins_cost,
        timeout=10.0
        )
        #upper_bound=20.0,
 
        return ged
            

