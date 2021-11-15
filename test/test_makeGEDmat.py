import pytest
import os,sys
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold 
import pickle
from joblib import Parallel, delayed
from PIL import Image
import networkx as nx

# Rdkit
from rdkit import Chem
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.app as app
from sphg.common.sphgviewer import sphgviewer
from sphg.infra import GraphEditDistanceSimilarity
from sphg.infra import xmol as xmol

# (1) SPhG
def ged_each(datadir,ver):

    with open(datadir+ver+'/'+'st03_res_selectSPhG.pickle', mode='rb') as f: res_selectSPhG_list3 = pickle.load(f)
    for SelectedNodeEdgeSPhGs in res_selectSPhG_list3[0:2]:#check Cov and NScf
        SelectedNodeSPhGs = SelectedNodeEdgeSPhGs[0]
        SelectedEdgeSPhGs = SelectedNodeEdgeSPhGs[1]
        nsphg = SelectedNodeSPhGs.shape[0]
        for isphg in range(nsphg):
            print(SelectedNodeSPhGs[isphg,:])
            print(SelectedEdgeSPhGs[isphg,:])
    
    SelectedNodeEdgeSPhGs = res_selectSPhG_list3[1]# NScf
    SelectedNodeSPhGs = SelectedNodeEdgeSPhGs[0]
    SelectedEdgeSPhGs = SelectedNodeEdgeSPhGs[1]
    ged0 = GraphEditDistanceSimilarity(datadir,ver)    
    #for isphg in range(nsphg):  ged0.calcVecEach(isphg,SelectedNodeSPhGs,SelectedEdgeSPhGs) 
    Parallel(n_jobs=-1, verbose=10)(delayed(ged0.calcVecEach)(isphg,SelectedNodeSPhGs,SelectedEdgeSPhGs) for isphg in range(nsphg) )
    return nsphg

def ged_merge(datadir,ver,nsphg):
    ged_mat = np.zeros((nsphg,nsphg),float)
    for isphg in range(nsphg):
        ged_vec = np.loadtxt(datadir + ver + '/' + 'ged_vec_%d.csv'%isphg, delimiter=',')
        ged_mat[isphg,:]=ged_vec.reshape((1,-1))

    np.savetxt(datadir + ver + '/' + 'ged_mat.csv', ged_mat, delimiter=',', fmt='%f')

#if __name__ == '__main__': 
def test_makeGEDsimilarity():

    tid = 104219 #int(sys.argv[1])
    datadir = './data/sphg/tid-%d/'%tid
    ntraintest = 0#int(sys.argv[3])
    scaffoldtype='bm'#sys.argv[6]#'ccrsingle' or 'ccrrecap' or 'bm'
    seed=99#int(sys.argv[2])
    ver = 't%ds%d%sntt1%d'%(tid,seed,scaffoldtype,ntraintest)

    # SPhG
    nsphg=ged_each(datadir,ver)
    ged_merge(datadir,ver,nsphg)
    
