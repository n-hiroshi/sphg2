# -*- coding: utf-8 -*-
"""
STEP3: Select the designated num of the most important SPhGs with Conerage, NScaffold, and Growth-rate criteria.
"""
import pytest
from rdkit import Chem
import os,sys
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
import pickle
from sphg.infra.select import select

def run(tid,ver,datadir,MineNodeSPhGs,MineEdgeSPhGs, MineHitvecs, SearchHitvecs_train_neg,nsphg=50):
    """
    STEP3: Select the designated num of the most important SPhGs with Conerage, NScaffold, and Growth-rate criteria.

    Args:
        tid: targed id
        ver: label of datasets
        datadir: data directory path
        MineNodeSPhGs: MineNodeSPhGs
        MineEdgeSPhGs: MineNodeSPhGs
        MineHitvecs: MineHitecs
        SearchHitvecs_train_neg: SearchHitvecs_train_neg (for calculating Growth-Rate)
        nsphg: nsphg

    Returns:
        SelectedNodeEdgeSPhGsCov
        SelectedNodeEdgeSPhGsScf
        SelectedNodeEdgeSPhGsGR

    """

    #(1) setup paths
    datadir_ver = datadir + ver + '/'
    rootname = ver + '_tid-%s-train-pos'%tid
    rootname_neg = ver + '_tid-%s-train-neg'%tid

    #(2) get train_pos_mols
    df = pd.read_csv(datadir_ver+'tid-%d-train-pos.csv'%tid)
    chembl_list_pos = df['chembl_id'].values
    scaffold_list_pos = df['scaffold'].values

    #(3) get train_neg_mols
    df = pd.read_csv(datadir_ver+'tid-%d-train-neg.csv'%tid)
    chembl_list_neg = df['chembl_id'].values
    scaffold_list_neg = df['scaffold'].values

    #(4) define selectSPhG
    select0 = select(MineNodeSPhGs,MineEdgeSPhGs)

    #(5) s1Cov select SPhGs with Coverage algorithm
    SelectedNodeEdgeSPhGsCov = select0.selectCov(MineHitvecs,nsphg)

    #(6) s2Scf select SPhGs with Nscaffold algorithm
    SelectedNodeEdgeSPhGsScf = select0.selectScf(MineHitvecs,scaffold_list_pos,nsphg)

    #(7) s3GR select SPhGs with Growth-rate algorithm
    # MineHitvecs means here SearchHitvecs_train_POS.
    # Growth-rate is the measure of ratios of No. of Hit in train_pos over No. of hit in train-neg 
    SelectedNodeEdgeSPhGsGR  = select0.selectGR(MineHitvecs,SearchHitvecs_train_neg,nsphg)

    #(8) return the results
    # SelectedNodeEdgeSPhGsXX is a list of [selectedNodeSPhGs, selectedEdgeSPhGs]
    return [SelectedNodeEdgeSPhGsCov,SelectedNodeEdgeSPhGsScf,SelectedNodeEdgeSPhGsGR]

