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

def read(tid,datadir):
    # actives
    datapath1=datadir + "tid-"+str(tid)+"-actives.txt"
    if ".txt" or ".tsv" in datapath1:
        df1 = pd.read_csv(datapath1, delimiter='\t')
        df1 = df1[['chembl-id','non_stereo_aromatic_smieles','pot.(log,Ki)']]
        df1 = df1.rename(columns={'pot.(log,Ki)': 'pot.(log,IC50)'})
        df1 = df1.rename(columns={'non_stereo_aromatic_smieles': 'nonstereo_aromatic_smiles'})
        smiles_list = df1['nonstereo_aromatic_smiles'].values
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
        activity_list = list(df1['pot.(log,IC50)'].values)
        chembl_list = list(df1['chembl-id'].values)
    # inactives
    datapath2=datadir + "tid-"+str(tid)+"-inactives.txt"
    if ".txt" or ".tsv" in datapath2:
        df2 = pd.read_csv(datapath2, delimiter='\t')
        df2 = df2[['chembl-id','nonstereo_aromatic_smiles']]
        smiles_list2 = df2['nonstereo_aromatic_smiles'].values
        smiles_list2 = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in smiles_list2]
        activity_list2 = list(np.zeros(len(smiles_list2)))
        chembl_list2 = list(df2['chembl-id'].values)
    # concatinate actives and inactives
    smiles_list.extend(smiles_list2)
    activity_list.extend(activity_list2)
    chembl_list.extend(chembl_list2)
    smiles_list,activity_list,chembl_list = molwt.screen(smiles_list,activity_list,chembl_list )
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



def getAllPhctsToXmolDB(imol):
    files=os.listdir(datadir+'xmoldbs/')
    picklefile=datadir+'xmoldbs/'+'xmoldb__tid-%d_id%06d'%(tid,imol)+'.pickle'
    picklefileshort='xmoldb__tid-%d_id%06d'%(tid,imol)+'.pickle'
    isfound=False

    print('')
    #datadir = './data/'
    #tid=8#11#194
    #datadir = '../../data/tid-%d/'%tid
    chembl_list, smiles_list, activity_list = read(tid,datadir)
    chembl = chembl_list[imol]
    smiles = smiles_list[imol]
    activity = activity_list[imol]
    nmol=len(chembl_list)
    print('### calc. %d-th/%d %s'%(imol,nmol,chembl))
    print(' SMILES = %s'%smiles)
    mol=Chem.MolFromSmiles(smiles)
    xmol0 = xmol(mol,molid=chembl)
    xmol0.activity=activity
    xmol0.getFeats()

    #xmol0.getAllPhcts(nCF=6,phcg=phcg)
    xmol0.statMultiMolSPhct()
 

def parallel_getALlPhcts():
    xmollist=[]
    #tid=8#11#194
    #datadir = '../../data/tid-%d/'%tid
    chembl_list, smiles_list, activity_list = read(tid,datadir)
    nmol=len(chembl_list)
    #nmol=16#nmol=len(chembl_list)
    ###################################
    #Parallel(n_jobs=4, verbose=10)( [delayed(getAllPhctsToXmolDB)(i) for i in range(nmol)] )    
    [getAllPhctsToXmolDB(i) for i in range(nmol)]    
    ###################################

    #picklefile=datadir+'xmollist_tid-%d'%tid+'.pickle'
    #with open(picklefile, mode='wb') as f:
    #    pickle.dump(xmollist, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
#def test_st101():
    global tid
    tid=int(sys.argv[1])
    global datadir
    #datadir = './data/phct/tid-%d/'%tid
    datadir = '../../data/phct/tid-%d/'%tid
    #datadir = '../../data/phcg/tid-%d/'%tid
    global phcg 
    phcg = True
    parallel_getALlPhcts()
    assert True
