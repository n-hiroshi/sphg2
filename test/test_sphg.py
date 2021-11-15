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
from sphg.core.graph import Graph
from itertools import combinations
from itertools import permutations
from sphg.common.sphgviewer import sphgviewer
from sphg.infra.canonicalize import Canonicalizer
    
# Set example data
smiles_list = [    'c1ccncc1',#pyridine
                   'CCc1ccccc1',#Me-Ph
                   'CNC(C(c1ccccc1)c2ccccc2)C(=O)N3CCCC3C(=O)NC(CCCNC(=N)N)C(=O)c4nc5ccccc5s4',#tid-11 CHRMBL403768 thrombin
                   "CN1CCN(CC1)CCCOC2=C(C=C3C(=C2)N=CC(=C3NC4=CC(=C(C=C4Cl)Cl)OC)C#N)OC",#Bosutinib
                   "CC1=C(C(=CC=C1)Cl)NC(=O)C2=CN=C(S2)NC3=NC(=NC(=C3)N4CCN(CC4)CCO)C",#Desatiinib
                   "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",#Imatinib
                   "CC1=C(C=C(C=C1)C(=O)NC2=CC(=C(C=C2)CN3CCN(CC3)C)C(F)(F)F)C#CC4=CN=C5N4N=CC=C5",#Potatinib
                   "CC1=C(C=C(C=C1)C(=O)NC2=CC(=CC(=C2)N3C=C(N=C3)C)C(F)(F)F)NC4=NC=CC(=N4)C5=CN=CC=C5",#Nilotibib
                   "CC(CC)(CCC)CC",#No ring
                   "C1=CC=C2C=CC=CC2=C1", # naphthalene
                   "Cc1ccc2[nH]ncc2c1-c1cc2cnc(NC(=O)C3CC3F)cc2cn1",
                   "Cc1cc(C(O)C(F)(F)F)ncc1-c1ccc2cc(NC(=O)C3CC3F)ncc2c1",
                   "O=C(Nc1cc2ccc(-c3cccc4[nH]nnc34)cc2cn1)C1CC1",
                   "CCSCc1ccc2c(c1)c1c3c(c4c5cc(CSCC)ccc5n5c4c1n2C1CC(O)(C(=O)OC)C5(C)O1)CNC3=O",
                   "CNC(=O)Nc1cc2ccc(-c3cc(F)ccc3C)cc2cn1",
                   "O=C1CCC2(O)C3Cc4ccc(O)cc4C2(CCN3CC2CC2)C1",#k-opioid
                   "COC(=O)C1CC(OC(C)=O)C(=O)C2C1(C)CCC1C(=O)OC(C(O)c3ccco3)CC12C"#k-opioid
                   "CNC(Cc1ccccc1)C(=O)N1CCCC1C(=O)NC(CCCN=C(N)N)C(=O)c1nc2ccc(OC)cc2s1"
              ]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def test_Graph_class():

    g = Graph(3)
    g.graph = [[0,1,0],
               [1,0,2],
               [0,2,0]]
    g.graphnp = np.asarray(g.graph)
    distvec,_=g.dijkstra(2)
    assert list(distvec) == [3,2,0]
    distvec,_=g.dijkstra(1)
    assert list(distvec) == [1,0,2]

def test_xmol():
    xmol0 = xmol(mols[0])
    assert xmol0.adjmat.tolist() ==[[0,1,0,0,0,1],[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[1,0,0,0,1,0]]

def _test_xmol_feats():
    xmol0 = xmol(mols[0])
    xmol0.getChemicalFeats()
    assert xmol0.featsAtomListGraph == [[3], [0, 1, 2, 3, 4, 5]]
    xmol1 = xmol(mols[1])
    xmol1.getChemicalFeats()
    assert xmol1.featsAtomListGraph == [[2,3,4,5,6,7],[0,1]]

    # thrombin inhibitor
    xmol2 = xmol(mols[2])
    xmol2.getChemicalFeats()
    print(xmol2.featsAtomListGraph)
    print(xmol2.featsFamilyList)
    assert xmol2.featsAtomListGraph == [[1], [18], [25], [30], [32], [33], [17], [24], [35], [37], [1], [30, 31, 32, 33], [36, 37, 38, 43, 44], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15], [38, 39, 40, 41, 42, 43], [0]]
    assert xmol2.featsFamilyList == [1,1,1,1,1,1,2,2,2,2,4,4,6,6,6,6,7]

    # Bosutinib
    xmol3 = xmol(mols[3])
    xmol3.getChemicalFeats()
    print(xmol3.featsAtomListGraph)
    print(xmol3.featsFamilyList)
    assert xmol3.featsAtomListGraph ==[[1], [4], [21], [10], [17], [30], [34], [1], [4], [11, 12, 13, 14, 15, 16], [14, 15, 17, 18, 19, 20], [22, 23, 24, 25, 26, 27], [0], [28], [29], [31], [35]]
    assert xmol3.featsFamilyList == [1, 1, 1, 2, 2, 2, 2, 4, 4, 6, 6, 6, 7, 7, 7, 7, 7]

def check_graph(distmat,featsListForAtom):
    pv0=sphgviewer()
    pv0.show_woDecode(distmat,featsListForAtom,figsize=(6,4))

def test_show_MolSPhG2():
    xmol3 = xmol(mols[2])
    xmol3.getChemicalFeats()

    sphcgList = xmol3.g.abstractGraph(xmol3.featsAtomListGraph,xmol3.featsFamilyList,verbose=True)
    sphcgList = xmol3.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: 
        print("#whole mole phcg")
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

def test_show_MolSPhG3():
    xmol3 = xmol(mols[3])
    xmol3.getChemicalFeats()

    sphcgList = xmol3.g.abstractGraph(xmol3.featsAtomListGraph,xmol3.featsFamilyList,verbose=True)
    sphcgList = xmol3.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: 
        print("#whole mole phcg")
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

def test_show_MolSPhG5():
    xmol3 = xmol(mols[5])
    xmol3.getChemicalFeats()

    sphcgList = xmol3.g.abstractGraph(xmol3.featsAtomListGraph,xmol3.featsFamilyList,verbose=True)
    sphcgList = xmol3.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: 
        print("#whole mole phcg")
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)


def _test_getAllSPhG3():
    xmol3 = xmol(mols[3])
    xmol3.getChemicalFeats()
    xmol3.getAllSPhGs()
    nSPhGs=len(xmol3.nodeSPhGs)

    sphcgList = xmol3.g.abstractGraph(xmol3.featsAtomListGraph,xmol3.featsFamilyList,verbose=True)
    sphcgList = xmol3.g.convertAromaticToBond(sphcgList)
    for sphcg0 in sphcgList: 
        print("#whole mole phcg")
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)

    for iSPhGs in range(nSPhGs):
        import common.sphgviewer
        pv0=sphgviewer()
        print("#sphcg %d"%iSPhGs)
        check_graph(sphcg0.distmat,sphcg0.featsListForAtom)
        #pv0.show(xmol3.nodeSPhGs[iSPhGs,:],xmol3.edgeSPhGs[iSPhGs,],figsize=(6,4))
        if iSPhGs>1: break

