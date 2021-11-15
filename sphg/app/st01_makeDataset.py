"""
STEP1: Data cleansing and buit data set for scaffold-hopping
Get assay data from curated data sets.
"""
import pytest
from rdkit import Chem
import os,sys,pickle
import random
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/'))
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.Scaffolds import MurckoScaffold
from sphg.common.unique import unique
from sphg.common.scaffold import Scaffold


def save_lists(filepath,chembl_list1,smiles_list1, activity_list1, scaffold_list1):
    nmol = len(chembl_list1)
    assert nmol == len(smiles_list1)
    assert nmol == len(activity_list1)
    assert nmol == len(scaffold_list1)
    print('##### nmol %d'%nmol)
    f = open(filepath,'w') 
    f.write('chembl_id,smiles,scaffold,activity\n')
    for imol in range(nmol):
        f.write(chembl_list1[imol].replace(',','_'))
        f.write(',')
        f.write(smiles_list1[imol])
        f.write(',')
        f.write(scaffold_list1[imol])
        f.write(',')
        f.write(str(activity_list1[imol]))
        f.write('\n')
    f.close()

def initUnique(chembl_list0,smiles_list0,activity_list0):
    chembl_list1=[]
    smiles_list1=[]
    activity_list1=[]
    for chembl0,smiles0, activity0 in zip(chembl_list0,smiles_list0,activity_list0):
        isMulti=False
        for chembl1 in chembl_list1:
            if chembl0==chembl1: 
                print(chembl0)
                isMulti=True
                assert False
        if not(isMulti):
            chembl_list1.append(chembl0)
            smiles_list1.append(smiles0)
            activity_list1.append(activity0)
    return chembl_list1, smiles_list1, activity_list1
        

def run(tid,ver,datadir,ntraintest=4,threshold=6.0,seed=1):
    ### (1) read required data
    os.makedirs(datadir + ver,exist_ok=True)

    picklefile=datadir+'xmoldbs_tid%d.pickle'%tid
    with open(picklefile, mode='rb') as f:
        xmoldbs = pickle.load(f)

    chembl_list_all=[]
    smiles_list_all=[]
    activity_list_all=[]
    for xmoldb in xmoldbs:
        chembl_list_all.append(xmoldb.chemblid)
        smiles_list_all.append(xmoldb.smiles)
        activity_list_all.append(xmoldb.activity)

    f=open(datadir+ver+'/'+'tid-%d-xmoldb-read.csv'%tid,'w')
    f.write('chembl-id,activity,smiles\n')
    for chembl,activity,smiles in zip(chembl_list_all,activity_list_all,smiles_list_all):
        f.write("%s,%s,%s\n"%(chembl,activity,smiles))
    f.close()

    chembl_list_all, smiles_list_all, activity_list_all = initUnique(chembl_list_all, smiles_list_all, activity_list_all)

    ### (2) assign scaffold to each elements of smiles_list_all
    scaffold0=Scaffold()
    scaffold0=Scaffold()
    scaffold_list_all = scaffold0.scaffold_smiles(smiles_list_all)

    print(len(smiles_list_all))
    print(len(activity_list_all))
    print(len(chembl_list_all))
    print(len(scaffold_list_all))
    nmol_xmoldb = len(smiles_list_all)

    ### (3) save scaffold to tid-x-all.csv
    file_list_all = datadir+ver+'/'+"tid-"+str(tid)+"-all.csv"
    save_lists(file_list_all,chembl_list_all,smiles_list_all, activity_list_all,scaffold_list_all)
    scaffold0 = Scaffold(file_list_all)
    scaffold_list_all=scaffold0.scaffold_smiles(smiles_list_all)
    unique_scaffold_list = unique.unique(scaffold_list_all)
    nunique_scaffold = len(unique_scaffold_list)
    index_unique_scaffold_list = range(nunique_scaffold)
    np.random.seed(seed)
    index_unique_scaffold_list = np.random.permutation(index_unique_scaffold_list)
    print("#seed: %d"%seed)
    print(index_unique_scaffold_list)
    print('num all scaffold: %d'%(len(index_unique_scaffold_list)))

    ### (4) split scaffolds to train and test  
    unique_scaffold_list_train = []
    unique_scaffold_list_test = []
    for index_unique_scaffold, scaffold in zip(index_unique_scaffold_list,unique_scaffold_list):
        if ntraintest >= 1:
            if index_unique_scaffold < int(nunique_scaffold/(ntraintest+1)):###################
                unique_scaffold_list_train.append(scaffold)
            else:
                unique_scaffold_list_test.append(scaffold)
        # the ntraintest=0 option is added for analyzeAllCompunds.py
        elif ntraintest == 0: 
            unique_scaffold_list_train.append(scaffold)
            unique_scaffold_list_test.append(scaffold)
             
    print('#unique_scaffold_list_train %d'%(len(unique_scaffold_list_train)))
    if ntraintest >= 1:
        print('#unique_scaffold_list_test %d'%(len(unique_scaffold_list_test)))

    ### (5) correct mols in scaffold_list_train/test_pos/neg
    # correct mols in scaffold_list_train_pos/neg
    smiles_list_train_pos=[]
    chembl_list_train_pos=[]
    activity_list_train_pos=[]
    scaffold_list_train_pos=[]
    smiles_list_train_neg=[]
    chembl_list_train_neg=[]
    activity_list_train_neg=[]
    scaffold_list_train_neg=[]
    #print(unique_scaffold_list_train)
    for scaffold in unique_scaffold_list_train:
        # get smiles-indeces belonging to each scaffold
        vec_index_scaffold  = [i for i, x in enumerate(scaffold_list_all) if x == scaffold]
        for index in vec_index_scaffold:
            if float(activity_list_all[index]) >= threshold:
                smiles_list_train_pos.append(smiles_list_all[index])
                chembl_list_train_pos.append(chembl_list_all[index])
                activity_list_train_pos.append(activity_list_all[index])
                scaffold_list_train_pos.append(scaffold_list_all[index])
            else:
                smiles_list_train_neg.append(smiles_list_all[index])
                chembl_list_train_neg.append(chembl_list_all[index])
                activity_list_train_neg.append(activity_list_all[index])
                scaffold_list_train_neg.append(scaffold_list_all[index])

    # correct mols in scaffold_list_test_pos/neg
    smiles_list_test_pos=[]
    chembl_list_test_pos=[]
    activity_list_test_pos=[]
    scaffold_list_test_pos=[]
    smiles_list_test_neg=[]
    chembl_list_test_neg=[]
    activity_list_test_neg=[]
    scaffold_list_test_neg=[]
    #print(unique_scaffold_list_test)
    for scaffold in unique_scaffold_list_test:
        # get smiles-indeces belonging to each scaffold
        vec_index_scaffold  = [i for i, x in enumerate(scaffold_list_all) if x == scaffold]
        for index in vec_index_scaffold:
            if float(activity_list_all[index]) >= threshold:
                smiles_list_test_pos.append(smiles_list_all[index])
                chembl_list_test_pos.append(chembl_list_all[index])
                activity_list_test_pos.append(activity_list_all[index])
                scaffold_list_test_pos.append(scaffold_list_all[index])
            else:
            #elif float(activity_list_all[index]) ==0.0:
                smiles_list_test_neg.append(smiles_list_all[index])
                chembl_list_test_neg.append(chembl_list_all[index])
                activity_list_test_neg.append(activity_list_all[index])
                scaffold_list_test_neg.append(scaffold_list_all[index])

 
    # cound num of scaffolds
    nmoltrain1=0
    nmoltest1=0
    molnumlist=[]
    for scaffold in unique_scaffold_list_train:
        vec_index_scaffold  = [i for i, x in enumerate(scaffold_list_all) if x == scaffold]
        #print(vec_index_scaffold)
        molnumlist+=vec_index_scaffold
        nmoltrain1+=len(vec_index_scaffold)
    for scaffold in unique_scaffold_list_test:
        vec_index_scaffold  = [i for i, x in enumerate(scaffold_list_all) if x == scaffold]
        #print(vec_index_scaffold)
        molnumlist+=vec_index_scaffold
        nmoltest1+=len(vec_index_scaffold)
    #print('nmoltrain1+nmoltest1 %d'%(nmoltrain1+nmoltest1))
    molnumlist.sort()

    # print num of mols in each list
    print('# num of train_pos: %d'%len(smiles_list_train_pos))
    print('# num of train_neg: %d'%len(smiles_list_train_neg))
    if ntraintest >= 1:
        print('# num of test_pos: %d'%len(smiles_list_test_pos))
        print('# num of test_neg: %d'%len(smiles_list_test_neg))
        print('# sum of pos: %d'%(len(smiles_list_train_pos)+len(smiles_list_test_pos)))
        print('# sum of neg: %d'%(len(smiles_list_train_neg)+len(smiles_list_test_neg)))
        assert len(smiles_list_train_pos)+len(smiles_list_test_pos)+len(smiles_list_train_neg)+len(smiles_list_test_neg)==nmol_xmoldb
    else:
        assert len(smiles_list_train_pos)+len(smiles_list_train_neg)==nmol_xmoldb
        assert len(smiles_list_test_pos)+len(smiles_list_test_neg)==nmol_xmoldb


    ### (7) save all lists
    save_lists(datadir+ver+'/'+"tid-"+str(tid)+"-train-pos.csv",chembl_list_train_pos,smiles_list_train_pos, activity_list_train_pos,scaffold_list_train_pos)
    save_lists(datadir+ver+'/'+"tid-"+str(tid)+"-train-neg.csv",chembl_list_train_neg,smiles_list_train_neg, activity_list_train_neg,scaffold_list_train_neg)
    if ntraintest >= 1:
        save_lists(datadir+ver+'/'+"tid-"+str(tid)+"-test-pos.csv",chembl_list_test_pos,smiles_list_test_pos, activity_list_test_pos,scaffold_list_test_pos)
        save_lists(datadir+ver+'/'+"tid-"+str(tid)+"-test-neg.csv",chembl_list_test_neg,smiles_list_test_neg, activity_list_test_neg,scaffold_list_test_neg)




