# -*- coding: utf-8 -*-
"""
Calculate all SPhGs for all molecules in a dataset and build a database of it as a file type of pickle
This is necessary before calculation of the performances with performanceCalculator.
"""
import pytest
from rdkit import Chem
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sphg.infra.xmol import xmol
import pickle
import itertools
import numpy as np
import pandas as pd
from sphg.common import pickle0
from sphg.common import molwt
from joblib import Parallel, delayed
from sphg.database.xmoldb import XmolDB, XmolDBHandler
from sphg.database import chembldb

class MacOSFile(object):
    """
    Utility class to increase the limit of the size of a pickle file
    """
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffer[idx:idx + batch_size])
            idx += batch_size

class DatabaseBuilder():
    """
    Build databases for actives and inactives retrieved from ChEMBEL
    """
    def getAllSPhGsToXmolDB(self,imol,tid,phg,datadir,chembl,smiles,activity):
        """
        Calculate SPhGs for a mol
        """
        files=os.listdir(datadir+'xmoldbs/')
        picklefile=datadir+'xmoldbs/'+'tid-%d_%s'%(tid,chembl)+'.pickle'
        picklefileshort='tid-%d_%s'%(tid,chembl)+'.pickle'

        isfound=False
        for file in files:
            if file == picklefileshort:
                isfound=True
        if isfound:
            print('### found. %d-th'%(imol))
        else:
            print('')
            #chembl = chembl_list2[imol]
            #smiles = smiles_list2[imol]
            #activity = activity_list2[imol]
            #nmol=len(chembl_list2)
            #print('### calc. %d-th/%d %s'%(imol,nmol,chembl))
            print('### calc. %d-th %s'%(imol,chembl))
            print(' SMILES = %s'%smiles)
            mol=Chem.MolFromSmiles(smiles)
            xmol0 = xmol(mol,molid=chembl)
            xmol0.activity=activity
            xmol0.getChemicalFeats()
            #xmol0.getAllSPhGs(nCF=6,verbose=True)
            #xmol0.getAllSPhGs(nCF=6,phg=phg) ### using infra.mpphct.py not infra.phct.py
            xmol0.getAllSPhGs(nCF=6,phg=phg) ### using infra.mpphct.py not infra.phct.py
     
            print('# %d-th mol'%imol)
            xmoldb0=XmolDB(chembl,smiles,activity)
            xmoldb0.nodeSPhGs = xmol0.nodeSPhGs.astype(np.int32)
            xmoldb0.edgeSPhGs = xmol0.edgeSPhGs.astype(np.int32)
     
            with open(picklefile, mode='wb') as f:
                pickle.dump(xmoldb0, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)
            #return xmol0

    def parallel_getALlSPhGs(self,tid,phg,datadir):
        """
        Mol-wise Parallelization of the calulation of SPhGs
        """
        xmollist=[]

        db0=chembldb()
        chembl_list2, smiles_list2, activity_list2 = db0.read(tid,datadir)
        f=open(datadir+'tid-%d-read.csv'%tid,'w')
        f.write('chembl-id,activity,smiles\n')
        for chembl,activity,smiles in zip(chembl_list2,activity_list2,smiles_list2): 
            f.write("%s,%s,%s\n"%(chembl,activity,smiles))
        f.close()

        nmol=len(chembl_list2)

        ###################################
        Parallel(n_jobs=-1, verbose=10)( [delayed(self.getAllSPhGsToXmolDB)(i,tid,phg,datadir,chembl_list2[i],smiles_list2[i],activity_list2[i]) for i in range(nmol)] )    
        ###################################


    def convert_to_XmolDB(self,tid,datadir):
        """
        Convert Xmol objects to XmolDB for supressing the data size 
        and make a single pickle file including all SPhGs of all mols.
        """
        XDBH = XmolDBHandler(tid,datadir)
        XDBH.convertXmolToXmolDB(datadir+'xmoldbs/')
        XDBH2 = XmolDBHandler(tid,datadir)
        XDBH2.readDB()


