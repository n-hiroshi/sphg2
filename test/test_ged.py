# -*- coding: utf-8 -*-
"""
Created on 2020/0620

@author: Hiroshi Nakano
"""
import pytest
from rdkit import Chem
import os,sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sphg.infra.xmol import xmol
from sphg.common.sphgviewer import sphgviewer
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
import copy
import itertools
from sphg.infra.graphEditDistanceSimilarity import GraphEditDistanceSimilarity

smiles_list = [
                   'CNC(C(c1ccccc1)c2ccccc2)C(=O)N3CCCC3C(=O)NC(CCCNC(=N)N)C(=O)c4nc5ccccc5s4',#tid-11 CHRMBL403768
                   "CN1CCN(CC1)CCCOC2=C(C=C3C(=C2)N=CC(=C3NC4=CC(=C(C=C4Cl)Cl)OC)C#N)OC",#Bosutinib
                   "CNC(=O)Nc1cc2ccc(-c3cc(Cl)ccc3C)cc2cn1",#kinase abl1
                   'CNC(C(c1ccccc1)c2ccccc2)C(=O)N3CCCC3C(C)NC(CCCNC(=N)N)C(C)c4nc5ccccc5s4',# modified tid-11 CHRMBL403768
              ]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def test_ged1():
    G=nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edges_from([(1, 2),(2,3)])
    

    G2=nx.Graph()
    G2.add_node(1)
    G2.add_node(2)
    G2.add_node(3)
    G2.add_edges_from([(1, 2),(1, 3)])
    ged1=nx.graph_edit_distance(G,G2)
    return ged1==2.0

def test_ged2():
    G=nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edges_from([(1, 2),(2,3)])
    

    G2=nx.Graph()
    G2.add_node(1)
    G2.add_node(2)
    G2.add_node(3)
    G2.add_edges_from([(1, 2),(1, 3)])

    def node_subst_cost(node1, node2):
        if node1 == node2:
            return 0
        return 1
    
    def node_del_cost(node):
        return 1  # here you apply the cost for node deletion
    
    def node_ins_cost(node):
        return 1  # here you apply the cost for node insertion
    
    def edge_subst_cost(edge1, edge2):
        if edge1==edge2:
            return 0
        return 1

    def edge_del_cost(node):
        return 2  # here you apply the cost for edge deletion
    
    def edge_ins_cost(node):
        return 2  # here you apply the cost for edge insertion

    ged2=nx.graph_edit_distance(
    G,
    G2,
    node_subst_cost=node_subst_cost,
    node_del_cost=node_del_cost,
    node_ins_cost=node_ins_cost,
    edge_subst_cost=edge_subst_cost,
    edge_del_cost=edge_del_cost,
    edge_ins_cost=edge_ins_cost
    )
    return ged2==4.0



def check_graph(distmat,featsListForAtom):
    pv0=sphgviewer()
    print(featsListForAtom)
    print(distmat)
    pv0.show_woDecode(distmat,featsListForAtom,figsize=(8,8))


def abstractSubGraph_forTest(xmol0,combiFeats):
    xmol0.getChemicalFeats()
    sphcgList = xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList,verbose=False)
    subSphcgList =  xmol0.g.abstractSubGraph(combiFeats,verbose=False)

    #for sphcg0 in subSphcgList:
    #    #sphcg0.print()
    #    print(sphcg0.featsListForAtom)
    #    check_graph(sphcg0.distmat,sphcg0.featsListForAtom)
    return subSphcgList


def test_graph6():
    print("\n\n------------------------------------")
    mol=mols[0]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    sphcgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    sphcgList = xmol6.g.convertAromaticToBond(sphcgList)
    #for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

    # graph1
    print("# graph1")
    combiFeats=(0, 1, 2, 3, 5, 6)
    SPhGList = abstractSubGraph_forTest(xmol6,combiFeats)
    sphg1 = SPhGList[0]
    print(sphg1.distmat)
    print(sphg1.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg1.distmat,sphg1.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print(nodesid)
    assert len(nodesid) == sphg1.distmat.shape[0]
    G1 = nx.from_numpy_matrix(sphg1.distmat)
    for inode, weight in enumerate(nodesid):
        G1.nodes[inode]['weight'] = weight

    G2 = nx.from_numpy_matrix(sphg1.distmat)
    for inode, weight in enumerate(nodesid):
        G2.nodes[inode]['weight'] = weight
    G2.nodes[0]['weight'] = 2.3
    #G2.nodes[1]['weight'] = 7
    #for node in G2.nodes: print(node['weight']+' ',end='')
    for i in range( len(nodesid)): print('%d '%G2.nodes[i]['weight'],end='')
    print()

    '''
    print("# graph2")
    combiFeats=(0, 1, 2, 3, 5, 7)
    SPhGList = abstractSubGraph_forTest(xmol6,combiFeats)
    sphg2 = SPhGList[0]
    print(sphg2.distmat)
    print(sphg2.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg2.distmat,sphg2.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print(nodesid)

    #
    assert len(nodesid) == sphg2.distmat.shape[0]
    G2 = nx.from_numpy_matrix(sphg2.distmat)
    for inode, weight in enumerate(nodesid):
        G2.nodes[inode]['weight'] = weight
    '''

    ###########################################################
    def node_subst_cost(node1, node2):
        #if node1['weight'] == node2['weight']:
        return np.abs(node1['weight'] - node2['weight'])

    
    def node_del_cost(node):
        return 1  # here you apply the cost for node deletion
    
    def node_ins_cost(node):
        return 1  # here you apply the cost for node insertion
    
    def edge_subst_cost(edge1, edge2):
        if edge1==edge2:
            return 0
        return 1

    def edge_del_cost(node):
        return 2  # here you apply the cost for edge deletion
    
    def edge_ins_cost(node):
        return 2  # here you apply the cost for edge insertion
    ##########################################################

    ged3=nx.graph_edit_distance(
    G1,
    G2,
    node_subst_cost=node_subst_cost,
    node_del_cost=node_del_cost,
    node_ins_cost=node_ins_cost,
    edge_subst_cost=edge_subst_cost,
    edge_del_cost=edge_del_cost,
    edge_ins_cost=edge_ins_cost
    )
    print(ged3)
    assert True

def test_graph7():
    print("\n\n------------------------------------")
    mol=mols[0]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    sphcgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    sphcgList = xmol6.g.convertAromaticToBond(sphcgList)
    #for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

    gedweightdict={'1':0,'2':1,'3':2,'4':3,'6':4,'8':5,'9':5,'12':6,'13':7,'14':8,'23':9,'24':10,'34':11,'123':12}
    # graph1
    print("# graph1")
    combiFeats=(0, 1, 2, 3, 5, 6)
    SPhGList = abstractSubGraph_forTest(xmol6,combiFeats)
    sphg1 = SPhGList[0]
    print(sphg1.distmat)
    print(sphg1.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg1.distmat,sphg1.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print(nodesid)
    assert len(nodesid) == sphg1.distmat.shape[0]
    G1 = nx.from_numpy_matrix(sphg1.distmat)
    for inode, weight in enumerate(nodesid):
        G1.nodes[inode]['weight'] = gedweightdict[str(weight)]

    print("# graph2")
    combiFeats=(0, 1, 2, 3, 5, 7)
    SPhGList = abstractSubGraph_forTest(xmol6,combiFeats)
    sphg2 = SPhGList[0]
    print(sphg2.distmat)
    print(sphg2.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg2.distmat,sphg2.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print(nodesid)

    #
    assert len(nodesid) == sphg2.distmat.shape[0]
    G2 = nx.from_numpy_matrix(sphg2.distmat)
    for inode, weight in enumerate(nodesid):
        G2.nodes[inode]['weight'] = gedweightdict[str(weight)]

    print("# graph3")
    combiFeats=(4, 5, 6, 7, 8, 9)
    SPhGList = abstractSubGraph_forTest(xmol6,combiFeats)
    sphg3 = SPhGList[0]
    print(sphg3.distmat)
    print(sphg3.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg3.distmat,sphg3.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print(nodesid)

    #
    assert len(nodesid) == sphg3.distmat.shape[0]
    G3 = nx.from_numpy_matrix(sphg3.distmat)
    for inode, weight in enumerate(nodesid):
        G3.nodes[inode]['weight'] = gedweightdict[str(weight)]

    ###########################################################
    node_subst_table = [[0,2,2,2,2,2, 1,1,1,2,2,2,1],#0    1=[0]
                        [2,0,2,2,2,2, 1,2,2,1,1,2,1],#1    2=[1]
                        [2,2,0,2,2,2, 2,2,1,2,1,2,1],#2    3=[2]
                        [2,2,2,0,2,2, 2,2,1,2,1,1,2],#3    4=[3]
                        [2,2,2,2,0,2, 2,2,2,2,2,2,2],#4    6=[5]
                        [2,2,2,2,2,0, 2,2,2,2,2,2,2],#5    9=[6]
                        [1,1,2,2,2,2, 0,2,2,2,2,2,2],#6   12=[0,1]
                        [1,2,1,2,2,2, 2,0,2,2,2,2,2],#7   13=[0,2]
                        [1,2,2,1,2,2, 2,2,0,2,2,2,2],#8   14=[0,3]
                        [2,1,1,2,2,2, 2,2,2,0,2,2,2],#9   23=[1,2]
                        [2,1,2,1,2,2, 2,2,2,2,0,2,2],#10  24=[1,3]
                        [2,2,1,1,2,2, 2,2,2,2,2,0,2],#11  34=[2,3]
                        [1,1,1,2,2,2, 2,2,2,2,2,2,0]]#12 123=[0,1,2]


    def node_subst_cost(node1, node2):
        return node_subst_table[node1['weight']][node2['weight']]

    
    def node_del_cost(node):
        if node['weight'] ==9:return 0.5 # junction node 9 = Serratosa's Carbon Link Node
        else: return 1.0
    
    def node_ins_cost(node):
        if node['weight'] ==9:return 0.5 # junction node 9 = Serratosa's Carbon Link Node
        else: return 1.0
    
    edgeLengthCostVec =[0.0] + [np.sum(1/np.arange(1,i+1)) for i in range(1,31)]
    print('edgeLengthCostVec')
    print(edgeLengthCostVec)
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

    ged2=nx.graph_edit_distance(
    G1,
    G2,
    node_subst_cost=node_subst_cost,
    node_del_cost=node_del_cost,
    node_ins_cost=node_ins_cost,
    edge_subst_cost=edge_subst_cost,
    edge_del_cost=edge_del_cost,
    edge_ins_cost=edge_ins_cost
    )
    print(ged2)

    ged3=nx.graph_edit_distance(
    G1,
    G3,
    node_subst_cost=node_subst_cost,
    node_del_cost=node_del_cost,
    node_ins_cost=node_ins_cost,
    edge_subst_cost=edge_subst_cost,
    edge_del_cost=edge_del_cost,
    edge_ins_cost=edge_ins_cost
    )
    print(ged3)


    ged0 = GraphEditDistanceSimilarity()
    ged4=ged0.ged(G1,G2)


    assert ged2==ged4


def test_ged_MolSPhG():
    print("\n\n------------------------------------")
    # graph1
    print("# graph1")
    mol=mols[0]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    molsphgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    molsphgList = xmol6.g.convertAromaticToBond(molsphgList)
    #for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

    gedweightdict={'1':0,'2':1,'3':2,'4':3,'6':4,'8':5,'9':5,'12':6,'13':7,'14':8,'23':9,'24':10,'34':11,'123':12}

    sphg1 = molsphgList[0]
    print(sphg1.distmat)
    print(sphg1.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg1.distmat,sphg1.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print('nodesid')
    print(nodesid)
    assert len(nodesid) == sphg1.distmat.shape[0]
    G1 = nx.from_numpy_matrix(sphg1.distmat)
    for inode, weight in enumerate(nodesid):
        G1.nodes[inode]['weight'] = gedweightdict[str(weight)]


    print("\n\n------------------------------------")
    # graph2
    print("# graph2")
    mol=mols[3]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    molsphgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    molsphgList = xmol6.g.convertAromaticToBond(molsphgList)
    #for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

    gedweightdict={'1':0,'2':1,'3':2,'4':3,'6':4,'8':5,'9':5,'12':6,'13':7,'14':8,'23':9,'24':10,'34':11,'123':12}

    sphg1 = molsphgList[0]
    print(sphg1.distmat)
    print(sphg1.featsListForAtom)
    nodesid,edgeid = xmol6.encodeSPhG(sphg1.distmat,sphg1.featsListForAtom)
    nodesid = nodesid[nodesid>0]
    print('nodesid')
    print(nodesid)
    assert len(nodesid) == sphg1.distmat.shape[0]
    G2 = nx.from_numpy_matrix(sphg1.distmat)
    for inode, weight in enumerate(nodesid):
        G2.nodes[inode]['weight'] = gedweightdict[str(weight)]




    ged0 = GraphEditDistanceSimilarity()
    ged4=ged0.ged(G1,G2)
    print(ged4)
    assert True


