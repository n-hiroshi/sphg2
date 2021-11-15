
import pytest
from rdkit import Chem
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import numpy as np
import pandas as pd
from sphg.common import molwt


class chembldb:
    """
    Database handler of chembl data set ( e.g. 'tid-11-master.csv')
    """
    def read(self,tid,datadir,activity_threshold=-1):

        datapath1=datadir + "tid-"+str(tid)+"-master.csv"
        #datapath1=datadir + "tid-"+str(tid)+"-read.csv"
        df1 = pd.read_csv(datapath1, delimiter=',')
        df1 = df1[['chembl-id','smiles','activity']]
        smiles_list = list(df1['smiles'].values)

        for smiles in smiles_list:
            print(smiles)
            dummy=Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            
        smiles_list = [Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) for smiles in smiles_list]
        activity_list = list(df1['activity'].values)
        chembl_list = list(df1['chembl-id'].values)

        smiles_list,activity_list,chembl_list = molwt.screen(smiles_list,activity_list,chembl_list,activity_threshold)
        chembl_list, smiles_list,activity_list = self.__unique(chembl_list,smiles_list,activity_list)

        return chembl_list, smiles_list, activity_list

    def __unique(self,chembl_list0,smiles_list0,activity_list0):
        chembl_list1=[]
        smiles_list1=[]
        activity_list1=[]
        for chembl0,smiles0, activity0 in zip(chembl_list0,smiles_list0,activity_list0):
            isMulti=False
            for chembl1 in chembl_list1:
                if chembl0==chembl1:
                    print(chembl0)
                    isMulti=True
                    #assert False
            for smiles1 in smiles_list1:
                if smiles0==smiles1:
                    print(smiles0)
                    isMulti=True
                    #assert False
            if not(isMulti):
                chembl_list1.append(chembl0)
                smiles_list1.append(smiles0)
                activity_list1.append(activity0)
        return chembl_list1, smiles_list1, activity_list1

