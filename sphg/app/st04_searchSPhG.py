# -*- coding: utf-8 -*-
"""
STEP4: Search test-molecules and background negatives (from pubchem or ZINC) with the selected SPhGs in STEP3.
In the STEP2, train-pos/neg mols are searched with all SPhGs generated from all train-pos.
On the other hand, this STEP searches test-pos/neg and backgroud negatives as well as train-pos/neg
only with the selected SPhGs in STEP3
"""
import pytest
from rdkit import Chem
import os,sys,pickle
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sphg.infra.miner import miner
from sphg.infra.search import search

def run(tid,ver,datadir,SelectedNodeEdgeSPhGs_list,nnegatives=1e+8):
    datadir_ver = datadir + ver +'/'
    os.makedirs(datadir_ver,exist_ok=True)

    ### (1) RETRIEFVE ALL xmoldbs 
    ## (1-1) RETRIEFVE ALL xmoldbs  for a specific tid
    xmoldbs_pickle = datadir + 'xmoldbs_tid%d.pickle'%tid
    with open(xmoldbs_pickle, mode='rb') as f:
        xmoldbs = pickle.load(f)

    ### (2) calc hitvec in order of train/test-pos/neg with (S)PhGs of Cov, Scf, GR selected in st03select
    SearchHitvecs_selectMethods = []
    SearchHitsmiles_selectMethods = []

    for iselectMethods,SelectedNodeEdgeSPhGs in enumerate(SelectedNodeEdgeSPhGs_list):
        print('\n### Search with the %d-th selectMethods(Cov, Scf, GR)'%(iselectMethods+1))

        ## (2-1) define and register SelectedNode/EdgeSPhGs in a search-instance
        #SelectedNodeSPhGs,SelectedEdgeSPhGs, priority = SelectedNodeEdgeSPhGs
        SelectedNodeSPhGs,SelectedEdgeSPhGs = SelectedNodeEdgeSPhGs
        search0 = search(SelectedNodeSPhGs,SelectedEdgeSPhGs)

        ## (2-2) prepare Allnode/edgesSPhGs for train-pos
        molSetName='train-pos'
        print('## Pharmacophore search in %s'%molSetName)
        
        ## (2-2-1) get all smiles in train-pos
        filename = 'tid-%d-%s.csv'%(tid,molSetName)
        df = pd.read_csv(datadir_ver+filename)
        chembl_list_pos = df['chembl_id'].values
        
        AllnodeSPhGs = []
        AlledgeSPhGs = []
        Allsmiles = []
        Allchemblid = []
        
        ## (2-2-2) put node/edgeSPhGs in Allnode/edgeSPhGs
        for chemblid in chembl_list_pos:
            for xmoldb in xmoldbs:
                if chemblid == xmoldb.chemblid:
                    #print(chemblid)
                    AllnodeSPhGs.append(xmoldb.nodeSPhGs)
                    AlledgeSPhGs.append(xmoldb.edgeSPhGs)
                    Allsmiles.append(xmoldb.smiles)
                    Allchemblid.append(xmoldb.chemblid)
                    break
        
        ## (2-5) search Hitvecs 
        SearchHitvecs = search0.search(AllnodeSPhGs,AlledgeSPhGs)

        ## (2-6) Save TrTP(train-pos), TP(test-pos), FP, and AP(all test positives) as dictionary 
        Hitlist = []
        Hitsmiles = []
        for SearchHitvecs_each in SearchHitvecs:
            for izmol,isHit in enumerate(SearchHitvecs_each):
                if isHit: Hitlist.append(izmol)
        #Hitlist=list(set(list(Hitlist)))
        for ifpzmol,izmol in enumerate(Hitlist):
            Hitsmiles.append(Allsmiles[izmol])
        Hitsmiles=list(set(Hitsmiles))

        SearchHitvecs_selectMethods.append(SearchHitvecs)
        SearchHitsmiles_selectMethods.append(Hitsmiles)# AP=All test-positives 
     
    ### (3) RETURN
    # return object is a list of three select methods with Cov, Scf, and GR,
    # each of which is a dictionary of 4 two-dim-numpy-array SearchHitvecs
    return SearchHitvecs_selectMethods,SearchHitsmiles_selectMethods


def printHitvec(hitvecs):
    nrow, ncol = hitvecs.shape
    for irow in range(nrow):
        [print(int(hitvecs[irow,icol]),end='')  for icol in range(ncol)] 
        print()


