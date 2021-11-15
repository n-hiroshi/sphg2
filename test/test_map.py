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
from adjustText import adjust_text

# Rdkit
from rdkit import Chem

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.app as app
from sphg.common.sphgviewer import sphgviewer
from sphg.infra import GraphEditDistanceSimilarity

class map():
    def __init__(self,datadir,ver,tid,nsphg):
        self.datadir=datadir
        self.ver=ver
        self.tid=tid
        self.nsphg=nsphg

    def getDistanceMat(self):

        #(1) retrieve performance
        with open(self.datadir+self.ver+'/'+'st04_%d_res_searchSPhG.pickle'%self.nsphg, mode='rb') as f:
            [SearchHitvecs_selectMethods,SearchHitsmiles_selectMethods] = pickle.load(f)
        print(SearchHitvecs_selectMethods[1])#NScf
        performance_vec = np.zeros(self.nsphg)

        nmol_train_pos = SearchHitvecs_selectMethods[1].shape[1]
        print('nmol_train_pos: %d'%nmol_train_pos)

        hitnum_train_pos = np.sum(SearchHitvecs_selectMethods[1],axis=1)
        print(hitnum_train_pos)

        color_sphg = hitnum_train_pos/nmol_train_pos # coverage

        for isphg, coverage in enumerate(color_sphg):
            print("SPhG%03d: %6.3f"%(isphg,coverage))    
            hitvec = SearchHitvecs_selectMethods[1][isphg,:]
            nmol = len(hitvec)
            for i,v in enumerate(hitvec): 
                print(int(v),end='')
                if (i+1)%10==0: print(' ',end='')
                if (i+1)%100==0: print()
            print()

        #(2) get ged mat
        ged_mat = np.loadtxt(self.datadir + self.ver + '/' + 'ged_mat.csv', delimiter=',')
        # to avoid non-symmetric-matrix error due to a self.very small differences of float values
        ged_mat = 0.5*(ged_mat+ged_mat.transpose())

        return ged_mat, color_sphg

    def drawMap(self,withLabels=False,phg=False):

        ged_mat,color_sphg = self.getDistanceMat()
        
        #(3) Mapping
        #n_neighbors=10
        #methods['Isomap'] = manifold.Isomap(n_neighbors, metric='precomputed')
        #methods['MDS'] = manifold.MDS(dissimilarity='precomputed')
        #methods['t-SNE'] = manifold.TSNE(metric='precomputed',random_state=1)
        method = manifold.TSNE(metric='precomputed',random_state=1)

        fig = plt.figure(figsize=(10, 8))
        plt.rcParams["font.size"] = 16
        Y = method.fit_transform(ged_mat)
        ax = fig.add_subplot(1,1,1)
        ax.axis('off')
        ax.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
        Ytsne_ged = Y.copy() 
        sc = ax.scatter(Y[:, 0], Y[:, 1], s=20, alpha=0.5, c=color_sphg, cmap='jet')
        plt.colorbar(sc)
 
        if withLabels:
            for i, _ in enumerate(Y[:,0]):
                if i==100: break # to avoid adjust_text optimization stacking
                plt_text=ax.text(Y[i, 0], Y[i, 1], s='%d'%i)
                texts.append(plt_text)
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            plt.savefig('isomap_infoClustering%d_%d.png'%(isim,self.tid))
            plt.clf()
        
        if phg: plt.savefig('phg_map_%d.png'%(self.tid))
        else:   plt.savefig('sphg_map_%d.png'%(self.tid))


#if __name__ == '__main__': 
def test_map():
    tid=104219#int(sys.argv[1])
    phg=False
    if phg: datadir = './data/phg/tid-%d/'%tid
    else: datadir = './data/sphg/tid-%d/'%tid
    threshold =6.0#float(sys.argv[2])
    if tid==104219: threshold = 5.0
    ntraintest = 0#int(sys.argv[3])
    scaffoldtype='bm'#sys.argv[6]#'ccrsingle' or 'ccrrecap' or 'bm'
    seed=99
    ver = 't%ds%d%sntt1%d'%(tid,seed,scaffoldtype,ntraintest)
    nsphg=300
    map0=map(datadir,ver,tid,nsphg)
    map0.drawMap(withLabels=False,phg=phg)
    

