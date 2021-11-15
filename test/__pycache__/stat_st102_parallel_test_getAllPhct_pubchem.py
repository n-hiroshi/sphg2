# -*- coding: utf-8 -*-
"""
Created on 2020/0620

@author: Hiroshi Nakano
"""
import pytest
from rdkit import Chem
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from infra.phct import xmol
import pickle
import itertools
import numpy as np
import pandas as pd
from common import pickle0
from common import molwt
from joblib import Parallel, delayed
from database.xmoldb import XmolDB, XmolDBHandler

#def run(tid,datadir):
#    pickle0.pickle_func(datadir + 'liganddb.pickle',makeLiganddb,tid,datadir)

def read(tid,datadir):
    datapath1=datadir + "tid-%d-pubchem-negatives.csv"%tid
    #datapath1=zincdir + "zinc12_rand_1000.tsv"
    if ".txt" or ".tsv" in datapath1:
        df1 = pd.read_csv(datapath1)
        df1 = df1[['cid','washed_nonstereo_aromatic_smiles']]
        df1 = df1.rename(columns={'cid': 'chembl-id'})
        df1 = df1.rename(columns={'washed_nonstereo_aromatic_smiles': 'smiles'})
        smiles_list = list(df1['smiles'].values)
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
        nmol = len(smiles_list)
        activity_list = [0.0 for i in range(nmol)]
        chembl_list = list(df1['chembl-id'].values)
    
    smiles_list,activity_list,chembl_list = molwt.screen(smiles_list,activity_list,chembl_list)
    print('Read pubchem negatives finished.')
    return chembl_list, smiles_list, activity_list


############################################################
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        #print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            #print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            #print("done.", flush=True)
            idx += batch_size
############################################################



def getAllPhctsToXmolDB(imol,chembl,smiles,activity):
    files=os.listdir(datadir+'pubchem_xmoldbs/')
    picklefile=datadir+'pubchem_xmoldbs/'+'xmoldb__pubchem%d_id%06d'%(tid,imol)+'.pickle'
    picklefileshort='xmoldb__pubchem%d_id%06d'%(tid,imol)+'.pickle'
    isfound=False
    #for file in files:
    #    if file == picklefileshort:
    #        isfound=True
    #if isfound:
    #    print('### found. %d-th'%(imol))
    #else:
    print('')
    #datadir = './data/'
    #tid=8#11#194
    #datadir = '../../data/tid-%d/'%tid
    #chembl_list, smiles_list, activity_list = read(tid,datadir)
    #chembl = chembl_list[imol]
    #smiles = smiles_list[imol]
    #activity = activity_list[imol]
    #nmol=len(chembl_list)
    #print('### calc. %d-th %s'%(imol,chembl))
    #print(' SMILES = %s'%smiles)
    mol=Chem.MolFromSmiles(smiles)
    xmol0 = xmol(mol,molid=chembl)
    xmol0.activity=activity
    xmol0.getFeats()
    #xmol0.getAllPhcts(nCF=6,phcg=phcg,verbose=False)
    xmol0.statMultiMolSPhct()
 
    #print('# %d-th mol'%imol)
    #xmoldb0=XmolDB(chembl,smiles,activity)
    #xmoldb0.nodePhcts = xmol0.nodePhcts.astype(np.int32)
    #xmoldb0.edgePhcts = xmol0.edgePhcts.astype(np.int32)
 
    #with open(picklefile, mode='wb') as f:
    #    pickle.dump(xmoldb0, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
    #return xmol0

def test_parallel_getAllPhcts():
    xmollist=[]
    #tid=8#11#194
    #datadir = '../../data/tid-%d/'%tid
    chembl_list, smiles_list, activity_list = read(tid,datadir)
    nmol=len(chembl_list)
    #nmol=16#nmol=len(chembl_list)
    ###################################
    #Parallel(n_jobs=8, verbose=10)( [delayed(getAllPhctsToXmolDB)(i) for i in range(nmol)] )    
    Parallel(n_jobs=6, verbose=10)( [delayed(getAllPhctsToXmolDB)(i,chembl_list[i],smiles_list[i],activity_list[i]) for i in range(nmol)] )    
    #[getAllPhctsToXmolDB(i) for i in range(nmol)]    
    ###################################

    #picklefile=datadir+'xmollist_tid-%d'%tid+'.pickle'
    #with open(picklefile, mode='wb') as f:
    #    pickle.dump(xmollist, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


if __name__=='__main__':
    global tid
    tid=int(sys.argv[1])
    global datadir
    global phcg 
    phcg = False
    if phcg:
        datadir = '../../data/phcg/tid-%d/'%tid
    else:
        datadir = '../../data/phct/tid-%d/'%tid
    test_parallel_getAllPhcts()
    #1convert_to_XmolDB()
