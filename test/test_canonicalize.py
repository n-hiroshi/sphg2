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
from sphg.infra.canonicalize import Canonicalizer
from itertools import permutations
import copy

smiles_list = [    'c1ccncc1',#pyridine
                   'Cc1ccc(C)cc1',#para-2Me-Ph
                   'C1=CC=C2C=CC=CC2=C1',#naphthalene
                   'c1cc(C)ccc1c2ccc(C)cc2',#Me-Ph-Ph-Me
                   'CC(CC)CC(CC(CC)CC)CC',#Tree
                   'CC(CC1)CC(CC(CC)CC)CC1',#Not Tree
                   'CNC(C(c1ccccc1)c2ccccc2)C(=O)N3CCCC3C(=O)NC(CCCNC(=N)N)C(=O)c4nc5ccccc5s4',#tid-11 CHRMBL403768
                   "CN1CCN(CC1)CCCOC2=C(C=C3C(=C2)N=CC(=C3NC4=CC(=C(C=C4Cl)Cl)OC)C#N)OC",#Bosutinib
                   "CNC(=O)Nc1cc2ccc(-c3cc(F)ccc3C)cc2cn1",#kinase abl1
                   "COC(=O)Nc1nc2ccc(C(=O)c3ccccc3)cc2[nH]1",#kinase abl #572
                   "O=C1CCC2(O)C3Cc4ccc(O)cc4C2(CCN3CC2CC2)C1",#kopioid
                   "COC(=O)Nc1ccc2c(c1)CCC(C(=O)N(C)C(CN1CCC(O)C1)c1ccccc1)O2",#kopioid
                   "Cc1c(F)cncc1-c1ccc2cc(NC(=O)C3CC3F)ncc2c1"
              ]
mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

def test_encodeSPhG1():
    xmol0 = xmol(mols[8])
    xmol0.getChemicalFeats()
    sphcgList =  xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList)
    Feats=[1,2,4,5,6,7]
    nodeSPhG0=[]
    edgeSPhG0=[]
    nCF=6
    for permFeats in permutations(Feats,nCF):
        #print('\npermFeats')
        #print(permFeats)
        subSphcgList =  xmol0.g.abstractSubGraph(permFeats)
        sphcg0=subSphcgList[0]
        nodeSPhG,edgeSPhG = xmol0.encodeSPhG(sphcg0.distmat,sphcg0.featsListForAtom,nCF,verbose=False)
        if len(nodeSPhG) == 0:
            nodeSPhG0 = nodeSPhG
            edgeSPhG0 = edgeSPhG
        else:
            for u,v in zip(nodeSPhG,nodeSPhG0): assert u==v
            for edge,edge0 in zip(edgeSPhG,edgeSPhG0):
                for u,v in zip(edge,edge0): assert u==v

def test_encodeSPhG2():
    xmol0 = xmol(mols[8])
    xmol0.getChemicalFeats()
    sphcgList =  xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList)
    Feats=[1,2,4,5,6,7]
    nodeSPhG0=[]
    edgeSPhG0=[]
    nCF=6
    for permFeats in permutations(Feats,nCF):
        #print('\npermFeats')
        #print(permFeats)
        subSphcgList =  xmol0.g.abstractSubGraph(permFeats)
        sphcg0=subSphcgList[0]
        nodeSPhG,edgeSPhG = xmol0.encodeSPhG(sphcg0.distmat,sphcg0.featsListForAtom,nCF,verbose=False)
        if len(nodeSPhG) == 0:
            nodeSPhG0 = nodeSPhG
            edgeSPhG0 = edgeSPhG
        else:
            for u,v in zip(nodeSPhG,nodeSPhG0): assert u==v
            for edge,edge0 in zip(edgeSPhG,edgeSPhG0):
                for u,v in zip(edge,edge0): assert u==v

def test_canonicalize1():
    xmol0 = xmol(mols[8])
    xmol0.getChemicalFeats()
    sphcgList =  xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList)
    Feats=[1,2,4,5,6,7]
    nodes0=[]
    edges0=[]
    nCF=6
    for permFeats in permutations(Feats,nCF):
        subSphcgList =  xmol0.g.abstractSubGraph(permFeats)
        sphcg0=subSphcgList[0]

        distmat=sphcg0.distmat
        featsListForAtom=sphcg0.featsListForAtom
        edges = [[u,v,distmat[u,v]] for u,distrow in enumerate(distmat) for v, dist in enumerate(distrow) if dist!=0 and u<v]
        nodes = featsListForAtom
        c = Canonicalizer()
        nodes, edges = c.canonicalize(nodes,edges)

        if len(nodes0) == 0:
            nodes0 = nodes
            edges0 = edges
        else:
            for u,v in zip(nodes,nodes0): assert u==v
            for edge,edge0 in zip(edges,edges0):
                for u,v in zip(edge,edge0): assert u==v


def test_canonicalize2():
    for imol in [7,8,9,10]:
        xmol0 = xmol(mols[imol])
        xmol0.getChemicalFeats()
        sphcgList =  xmol0.g.abstractGraph(xmol0.featsAtomListGraph,xmol0.featsFamilyList)
        Feats=[1,2,4,5,6,7]
        nodes0=[]
        edges0=[]
        nCF=6
        #for perm in permutations(range(nCF),nCF):
        #    permFeats = [Feats[v] for v in perm]
 
        subSphcgList =  xmol0.g.abstractSubGraph(Feats)
        sphcg0=subSphcgList[0]
 
        distmat=sphcg0.distmat
        featsListForAtom=sphcg0.featsListForAtom
        edges = [[u,v,distmat[u,v]] for u,distrow in enumerate(distmat) for v, dist in enumerate(distrow) if dist!=0 and u<v]
        nodes = featsListForAtom
        c = Canonicalizer()
 
        #print(nodes)
        #print(edges)
 
        #permutation
        nodes0=[]
        edges0=[]
        nnodes = len(nodes)

        for perm in permutations(range(nnodes),nnodes):

            # perm = [2,0,1]
            # invperm = [1,2,0]
            # nodes =  [[3],[4],[6]]
            # edges =  [[0,1,5],[0,2,6],[1,2,32]]
            # nodes1 = [[6],[3],[4]]
            # edges1 = [[1,2,5],[1,0,6],[2,0,32]]

            invperm = [0 for v in nodes]
            for i,v in enumerate(perm): invperm[v] = i
 
            nodes1 = [nodes[v] for v in perm] #nodesはpermの順番で入れ替え
            edges1 = [[] for i in edges]
            for j,edge in enumerate(edges):
                for i,v in enumerate(edge):
                    if i<2:
                        edges1[j].append(invperm[v]) #edgesはpermの反対順invpermの順番で入れ替え
                    else:
                        edges1[j].append(v)
 
            #print(perm)
            #print(invperm)
            print('nodes1 before canocnicalization')
            print(nodes1)
            print('edges1 before canocnicalization')
            print(edges1)
            nodes1, edges1 = c.canonicalize(nodes1,edges1)
            print('nodes1 after canocnicalization')
            print(nodes1)
            print('edges1 after canocnicalization')
            print(edges1)
  
            if len(nodes0) == 0:
                nodes0 = nodes1
                edges0 = edges1
            else:
                for u,v in zip(nodes1,nodes0): assert u==v
                for edge1,edge0 in zip(edges1,edges0):
                    for u,v in zip(edge1,edge0): assert u==v










