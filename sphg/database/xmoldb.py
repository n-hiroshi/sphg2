"""
This library xmoldb.py provides the interface of the calculated databases stored as pickle files
"""
# -*- coding: utf-8 -*-
import os,sys,pickle
from rdkit import Chem
import numpy as np
import pytest
import itertools
import pandas as pd
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Scaffolds import MurckoScaffold
from sphg.infra.xmol import xmol
from sphg.core.graph import Graph
from sphg.common.unique import unique
from sphg.common.scaffold import Scaffold
from sphg.common import pickle0
import sphg.common.molwt as molwt
from sphg.database.chembldb import chembldb

class XmolDB():
    """
    An instance of this class has a smiles-string, an activity-value, a mol-instance of RDKit, 
    the Mol-SPhG and SPhG of the mol.
    Just To pickle all of the instances generated from the molecules of some dataset into an All-In-One-database of the target tid.
    """
    def __init__(self,chemblid,smiles,activity):
        self.chemblid=chemblid
        self.smiles=smiles
        self.activity=activity
        self.mol=Chem.MolFromSmiles(smiles)
        self.nodeSPhGs=None
        self.edgeSPhGs=None

class XmolDBHandler():
    def __init__(self,tid,datadir):
        self.datadir=datadir
        if tid>=0:
            self.tid=tid
            dbname='xmoldbs_tid%d.pickle'%tid
        self.picklefile=datadir+dbname

    def writeDB(self):
        with open(self.picklefile, mode='wb') as f:
            pickle.dump(self.xmoldbs,f)

    def readDB(self):
        with open(self.picklefile, mode='rb') as f:
            self.xmoldbs = pickle.load(f)

    def getAllSPhGs(self,chemblid,nCF):
        nodeSPhGs = [self.xmoldbs[chemblid].AllNodeSPhGs for xmoldb in self.xmoldbs if xmoldb.chemblid==chemblid]
        edgeSPhGs = [self.xmoldbs[chemblid].AllEdgeSPhGs for xmoldb in self.xmoldbs if xmoldb.chemblid==chemblid]
        return nodeSPhGs, edgeSPhGs

    def getNmol(self):
        return len(self.xmoldbs)

    def getMols(self,chembl_list):
        mols=[]
        for chemblid in chembl_list:
            _ = [mols.append(xmoldb) for xmoldb in self.xmoldbs if xmoldb.chemblid==chemblid]
        return mols

    def getActivities(self,chembl_list):
        activity_array=[]
        for chemblid in chembl_list:
            _ = [activity_array.append(xmoldb.activity_val) for xmoldb in self.xmoldbs if xmoldb.chemblid==chemblid]
        return activity_array
    def convertXmolToXmolDB(self,xmoldbsdir):
        # (1) retrieve xmoldb calculated by sphg.py
        files = os.listdir(xmoldbsdir)
        xmoldbs_temp = []
        for file in files:
            #print("# test %s"%file)
            if '.pickle' in file:
                print("# found %s"%file,flush=True)
                with open(xmoldbsdir+file, mode='rb') as f:
                    xmoldbs_temp.append(pickle.load(f))

        chembl_list, smiles_list, activity_list = self.readOriginalTXT(self.tid,self.datadir) 
        kmol=0
        self.xmoldbs = []
        for imol, (chemblid, smiles, activity) in enumerate(zip(chembl_list, smiles_list, activity_list)): 
            print("# checked %d-th mol"%imol,flush=True)
            isfound=False
            for jmol, xmoldb0 in enumerate(xmoldbs_temp):
                    if chemblid == xmoldb0.chemblid:
                         kmol+=1
                         self.xmoldbs.append(xmoldb0)
                         isfound=True
                         break
            if isfound==False:
               print('#WARNNING  %d-th mol not found, id=%s'%(imol,chemblid))
        print('# total %d mols found'%kmol)
        # (2) save xmoldbs with self.writeDB  
        self.writeDB()

    def readOriginalTXT(self,tid,datadir):
        chembl0=chembldb()
        chembl_list,smiles_list,activity_list = chembl0.read(tid,datadir)
        return chembl_list, smiles_list, activity_list

    


