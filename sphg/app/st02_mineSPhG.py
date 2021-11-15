# -*- coding: utf-8 -*-
"""
STEP2: list up all SPhGs from train-pos dataset and examine which mols of train-pos include which SPhGs  
"""
import pytest
from rdkit import Chem
import os,sys,pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from sphg.infra.miner import miner
from sphg.infra.search import search

def run(tid,ver,datadir):
    datadir_ver = datadir + ver +'/'
    os.makedirs(datadir_ver,exist_ok=True)

    ### (1) RETRIEFVE ALL xmoldbs 
    xmoldbs_pickle = datadir + 'xmoldbs_tid%d.pickle'%tid
    with open(xmoldbs_pickle, mode='rb') as f:
        xmoldbs = pickle.load(f)

    ### (2) MINING with train-pos
    ## (2-1) choose xmoldbs only in train-pos
    filename = 'tid-%d-train-pos.csv'%tid
    rootname = ver + '_tid-%s-train-pos'%tid

    df = pd.read_csv(datadir_ver+filename)
    chembl_list_pos = df['chembl_id'].values

    AllnodeSPhGs_train_pos = []
    AlledgeSPhGs_train_pos = []

    ## (2-2) put node/edgeSPhGs in Allnode/edgeSPhGs_train_pos for miner-instance 
    ## DON'T DO LIKE THE FOLLOWING 2 LINES because hitvec and the tidx-x-train-pos have different mol orders
    #for xmoldb in xmoldbs:
    #    if xmoldb.chemblid in chembl_list_pos:
    for chemblid in chembl_list_pos:
        for xmoldb in xmoldbs:
            if chemblid == xmoldb.chemblid:
                print(chemblid)
                AllnodeSPhGs_train_pos.append(xmoldb.nodeSPhGs)
                AlledgeSPhGs_train_pos.append(xmoldb.edgeSPhGs)
                break
    
    
    ## (2-3) mining
    miner0 = miner(AllnodeSPhGs_train_pos,AlledgeSPhGs_train_pos)
    MineNodeSPhGs, MineEdgeSPhGs, MineHitvecs = miner0.mine()


    ### (3) SEARCH with train-neg for calculating Growth-rate
    ## (3-1) choose xmoldbs only in train-neg
    filename = 'tid-%d-train-neg.csv'%tid
    rootname = ver + '_tid-%s-train-neg'%tid

    df = pd.read_csv(datadir_ver+filename)
    chembl_list_pos = df['chembl_id'].values

    AllnodeSPhGs_train_neg = []
    AlledgeSPhGs_train_neg = []

    ## (3-2) put node/edgeSPhGs in Allnode/edgeSPhGs_train_neg
    for chemblid in chembl_list_pos:
        for xmoldb in xmoldbs:
            if chemblid == xmoldb.chemblid:
                print(chemblid)
                AllnodeSPhGs_train_neg.append(xmoldb.nodeSPhGs)
                AlledgeSPhGs_train_neg.append(xmoldb.edgeSPhGs)
                break


    ## (3-3) search Hitvecs for tran-neg in order to calculate Growth-Rate
    search0 = search(MineNodeSPhGs,MineEdgeSPhGs)
    SearchHitvecs_train_neg = search0.search(AllnodeSPhGs_train_neg,AlledgeSPhGs_train_neg)
    return MineNodeSPhGs, MineEdgeSPhGs, MineHitvecs, SearchHitvecs_train_neg


