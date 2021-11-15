# -*- coding: utf-8 -*-
"""
Created on 2020/0620

@author: Hiroshi Nakano
"""
import pytest
from rdkit import Chem
import os,sys
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(__file__), '../sphg/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sphg.infra.xmol import xmol
from sphg.common.sphgviewer import sphgviewer
from itertools import combinations
from sphg.core.graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
import copy
import itertools

    
# Set example data
smiles_list = [    'c1ccncc1',#pyridine
                   'Cc1ccc(C)cc1',#para-2Me-Ph
                   'C1=CC=C2C=CC=CC2=C1',#naphthalene
                   'c1cc(C)ccc1c2ccc(C)cc2',#Me-Ph-Ph-Me
                   'CC(CC)CC(CC(CC)CC)CC',#Tree
                   'CC(CC1)CC(CC(CC)CC)CC1',#Not Tree
                   'CNC(C(c1ccccc1)c2ccccc2)C(=O)N3CCCC3C(=O)NC(CCCNC(=N)N)C(=O)c4nc5ccccc5s4',#tid-11 CHRMBL403768
                   "CN1CCN(CC1)CCCOC2=C(C=C3C(=C2)N=CC(=C3NC4=CC(=C(C=C4Cl)Cl)OC)C#N)OC",#Bosutinib
                   "CNC(=O)Nc1cc2ccc(-c3cc(Cl)ccc3C)cc2cn1",#kinase abl1
                   "COC(=O)Nc1nc2ccc(C(=O)c3ccccc3)cc2[nH]1",#kinase abl #572
                   "Cc1ccc2[nH]ncc2c1-c1cc2cnc(NC(=O)C3CC3F)cc2cn1",#kinase
                   "O=C1CCC2(O)C3Cc4ccc(O)cc4C2(CCN3CC2CC2)C1",#kopioid
                   "COC(=O)Nc1ccc2c(c1)CCC(C(=O)N(C)C(CN1CCC(O)C1)c1ccccc1)O2",#kopioid
                   "COC12CCC3(CC1C(O)CC1CCCCC1)C1Cc4ccc(O)c5c4C3(CCN1CC1CC1)C2O5",#kopioid
                   "Cc1c(F)cncc1-c1ccc2cc(NC(=O)C3CC3F)ncc2c1",
                   "C#CCN1CCC23CCCCC2C1Cc1ccc(O)cc13",
                   "CNC(Cc1ccccc1)C(=O)N1CCCC1C(=O)NC(CCCN=C(N)N)C(=O)c1nc2ccc(OC)cc2s1",
                   "COc1ccc2cc(S(=O)(=O)NC(Cc3ccc(CN)cc3)C(=O)N(C)C3CCCC3)ccc2c1",
"COc1ccc2cc(-c3nn(C(C)C)c4ncnc(N)c34)ccc2c1",
"C#CCN1CCC23CCCCC2C1Cc1ccc(O)cc13",
"C1=CC2=C3C(=C1)C4=CC=CC5=C4C(=CC=C5)C3=CC=C2", #perylene
"c1ccc2c3nc(nc4[nH]c(nc5nc(nc6[nH]c(n3)c7ccccc67)c8ccccc58)c9ccccc49)c2c1",#phthalocyanine
"CC(C)C1=C(C(=CC=C1)C(C)C)N2C(=O)C3=CC(=C4C5=C(C=C6C7=C5C(=C(C=C7C(=O)N(C6=O)C8=C(C=CC=C8C(C)C)C(C)C)OC9=CC=CC=C9)C1=C(C=C(C3=C41)C2=O)OC1=CC=CC=C1)OC1=CC=CC=C1)OC1=CC=CC=C1", #lumogen red
"COC1=CC=C(C=C1)N(C2=CC=C(C=C2)OC)C3=CC4=C(C=C3)C5=C(C46C7=C(C=CC(=C7)N(C8=CC=C(C=C8)OC)C9=CC=C(C=C9)OC)C1=C6C=C(C=C1)N(C1=CC=C(C=C1)OC)C1=CC=C(C=C1)OC)C=C(C=C5)N(C1=CC=C(C=C1)OC)C1=CC=C(C=C1)OC",#spiro-OMeTAD
"C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])O)O)N",#ATP
"CC[C@@H]1[C@@]([C@@H]([C@H](C(=O)[C@@H](C[C@@]([C@@H]([C@H]([C@@H]([C@H](C(=O)O1)C)O[C@H]2C[C@@]([C@H]([C@@H](O2)C)O)(C)OC)C)O[C@H]3[C@@H]([C@H](C[C@H](O3)C)N(C)C)O)(C)O)C)C)O)(C)O",#erythromycin
"O[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@]([H])(CO)O1",#glucose
"C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23",#C60
              "N=C(N)NCCCC1NC(=O)C2CCCN2C(=O)C(Cc2ccccc2)NC(=O)CCCCCCCCCNC(=O)C1=O",#thrombin macrocyclic
              "COc1ccc(OC)c(S(=O)(=O)NC2CCCCN(CC(=O)NCc3ccc(C(=N)N)cc3)C2=O)c1",
              "N=C(N)NCCCC1NC(=O)C2CCCN2C(=O)C(CCc2ccccc2)NC(=O)CCCCCCCNC(=O)C1=O",
              "N=C(N)NCCCC1NC(=O)C2CCCN2C(=O)C(Cc2ccccc2)NC(=O)CCCCCNC(=O)C1=O",
              "C=C(C)C1Cc2c(ccc3c2OC2COc4cc(OC)c(OC)cc4C2C3=O)O1",
              "COc1ccc2cc(CCNC(=O)c3ccc4c(c3)C3(C)CCN(CC5CC5)C(C4)C3C)ccc2c1",
"COc1nc(-c2ccccc2Cl)cc2cnc(NC(=O)C3CC3)cc12"
              ]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def test_Graph_class():

    g = Graph(3)
    graph = [[0,1,0],
               [1,0,2],
               [0,2,0]]
    g.graphnp = np.asarray(graph)
    distvec,_=g.dijkstra(2)
    assert list(distvec) == [3,2,0]
    distvec,_=g.dijkstra(1)
    assert list(distvec) == [1,0,2]

def test_xmol():
    xmol0 = xmol(mols[0])
    assert xmol0.adjmat.tolist() ==[[0,1,0,0,0,1],[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[1,0,0,0,1,0]]


def test_dijdstra1():
    xmol1 = xmol(mols[1])
    g = Graph(xmol1.mol.GetNumAtoms())
    assert g.V==8
    xmol0 = xmol(mols[0])
    g = Graph(xmol0.mol.GetNumAtoms())
    assert g.V==6
    g.graphnp=xmol0.adjmat
    distvec,_ = g.dijkstra(0)
    assert list(distvec) == [0,1,2,3,2,1]
   

def test_dijdstra2():
    xmol1 = xmol(mols[1])
    g = Graph(xmol1.mol.GetNumAtoms())
    g.graphnp=xmol1.adjmat
    distvec,prev = g.dijkstra(0)
    assert list(distvec) == [0,1,2,3,4,5,3,2]
    list_prev = g.del_m1_and_convertlist(prev,2)
    assert list_prev == [[],[0],[1],[2],[3,6],[4],[7],[1]]


def test_dijdstra3():
    xmol2 = xmol(mols[2])
    g = Graph(xmol2.mol.GetNumAtoms())
    g.graphnp=xmol2.adjmat
    distvec,prev = g.dijkstra(0)
    list_prev = g.del_m1_and_convertlist(prev,2)
    assert list(distvec) == [0,1,2,3,4,5,4,3,2,1]
    assert list_prev == [[], [0], [1], [2, 8], [3], [4, 6], [7], [8], [9], [0]]

def test_shortestpaths1():
    xmol1 = xmol(mols[1])
    g = Graph(xmol1.mol.GetNumAtoms())
    g.graphnp=xmol1.adjmat
    distvec,paths = g.shortestPaths(0)
    list_paths = g.del_m1_and_convertlist(paths,3)
    print(distvec)
    assert list(distvec) == [0,1,2,3,4,5,3,2]
    assert list_paths == [[[0]], [[0, 1]], [[0, 1, 2]], [[0, 1, 2, 3]], [[0, 1, 2, 3, 4], [0, 1, 7, 6, 4]], [[0, 1, 2, 3, 4, 5], [0, 1, 7, 6, 4, 5]], [[0, 1, 7, 6]], [[0, 1, 7]]]


def test_shortestpaths2():
    xmol1 = xmol(mols[3])
    g = Graph(xmol1.mol.GetNumAtoms())
    g.graphnp=xmol1.adjmat
    distvec,paths = g.shortestPaths(3)
    list_paths = g.del_m1_and_convertlist(paths,3)
    assert list(distvec) == [3,2,1,0,2,3,4,5,6,7,8,9,7,6]
    assert list_paths == [[[3, 2, 1, 0]], [[3, 2, 1]], [[3, 2]], [[3]], [[3, 2, 4]], [[3, 2, 4, 5]], [[3, 2, 1, 0, 6], [3, 2, 4, 5, 6]], [[3, 2, 1, 0, 6, 7], [3, 2, 4, 5, 6, 7]], [[3, 2, 1, 0, 6, 7, 8], [3, 2, 4, 5, 6, 7, 8]], [[3, 2, 1, 0, 6, 7, 8, 9], [3, 2, 4, 5, 6, 7, 8, 9]], [[3, 2, 1, 0, 6, 7, 8, 9, 10], [3, 2, 4, 5, 6, 7, 8, 9, 10], [3, 2, 1, 0, 6, 7, 13, 12, 10], [3, 2, 4, 5, 6, 7, 13, 12, 10]], [[3, 2, 1, 0, 6, 7, 8, 9, 10, 11], [3, 2, 4, 5, 6, 7, 8, 9, 10, 11], [3, 2, 1, 0, 6, 7, 13, 12, 10, 11], [3, 2, 4, 5, 6, 7, 13, 12, 10, 11]], [[3, 2, 1, 0, 6, 7, 13, 12], [3, 2, 4, 5, 6, 7, 13, 12]], [[3, 2, 1, 0, 6, 7, 13], [3, 2, 4, 5, 6, 7, 13]]]

def test_isTree1():
    xmol1 = xmol(mols[1])
    g = Graph(xmol1.mol.GetNumAtoms())
    g.graphnp=xmol1.adjmat
    assert g.isTree() ==False

    xmol3 = xmol(mols[3])
    g = Graph(xmol3.mol.GetNumAtoms())
    g.graphnp=xmol3.adjmat
    assert g.isTree() ==False

    xmol4 = xmol(mols[4])
    g = Graph(xmol4.mol.GetNumAtoms())
    g.graphnp=xmol4.adjmat
    assert g.isTree() ==True

    xmol5 = xmol(mols[5])
    g = Graph(xmol5.mol.GetNumAtoms())
    g.graphnp=xmol5.adjmat
    assert g.isTree() ==False

    xmol6 = xmol(mols[6])
    g = Graph(xmol6.mol.GetNumAtoms())
    g.graphnp=xmol6.adjmat
    assert g.isTree() ==False

def check_graph(distmat,featsListForAtom):
    pv0=sphgviewer()
    print(featsListForAtom)
    print(distmat)
    pv0.show_woDecode(distmat,featsListForAtom,figsize=(8,8))


def abstractSubGraph_forTest(xmol0,combiFeats):
    xmol0.getChemicalFeats()
    sphcgList = xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList,verbose=False)
    subSphcgList =  xmol0.g.abstractSubGraph(combiFeats,verbose=False)

    for sphcg0 in subSphcgList: 
        #sphcg0.print()
        print(sphcg0.featsListForAtom)
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)


def test_graph6():
    print("\n\n------------------------------------")
    mol=mols[8]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    sphcgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    sphcgList = xmol6.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)
    combiFeats=(0, 1, 2, 3, 5, 6)

    abstractSubGraph_forTest(xmol6,combiFeats)
    assert True

def _test_graph8():
    print("\n\n------------------------------------")
    mol=mols[8]
    #Chem.Draw.ShowMol(mol)
    xmol6 = xmol(mol)
    xmol6.getChemicalFeats()
    sphcgList = xmol6.g.abstractGraph(xmol6.featsAtomListGraph,xmol6.featsFamilyList,verbose=True)
    sphcgList = xmol6.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: check_graph(sphcg0.distmat,sphcg0.featsListForAtom)
   
    nfeats=7#len(xmol6.featsFamilyList)
    #print(nfeats)
    #print(xmol6.featsFamilyList)
    #print("------------------------------------")
    nCF=6
    #combiFeats=(1,2,3,5,6)
    #abstractSubGraph_forTest(xmol6,combiFeats)
    for i,combiFeats in enumerate(combinations(list(range(nfeats)),nCF)):
        print("------------------------------------")
        print(combiFeats)
        abstractSubGraph_forTest(xmol6,combiFeats)
    assert True


