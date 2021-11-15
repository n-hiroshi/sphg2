# -*- coding: utf-8 -*-
"""
Convert molecular structures to its scaffold with Bemis-Marko, CCR Single, and CCR Recap
"""
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import numpy as np

class Scaffold: 
    def __init__(self,filename=""):
        self.filename = filename
        if len(filename) > 0:
            self.df = pd.read_csv(filename)

    def get(self,smiles):
        return self.df[self.df['smiles']==smiles].iloc[0,2]

    def scaffold_mols(self,list_in):
        if len(self.filename) > 0:
            smiles_list = [Chem.MolToSmiles(mol) for mol in list_in]
            list_out=[self.get(smiles) for smiles in smiles_list]
        else:
            mols = list_in
            list_out=[ Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol)) for mol in mols]
        return list_out

    def scaffold_smiles(self,list_in):
        if len(self.filename) > 0:
            list_out=[self.get(smiles) for smiles in list_in]
        else:
            #smiles = list_in
            #list_out=[ Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smile))) for smile in smiles]
            list_out = []
            for smiles in list_in:
                scaffold1 = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(smiles)))
                #print(scaffold1)
                if len(scaffold1) ==0: list_out.append(smiles)
                else: list_out.append(scaffold1)
                #if isinstance(scaffold1,str):
                #    list_out.append( scaffold1 )
                #elif isinstance(scaffold1,float) and isnan(scaffold1):
                #    list_out.append( smiles )
                #    print(smiles)
                #else: assert False

        return list_out

    def scaffold_smiles_ccr(self,ccrfile,list_in):
        df_ccr = pd.read_csv(ccrfile,delimiter='|',header=None,index_col=2)
        ccr_list = []
        for chemblid in list_in:
            try:
                ccr = df_ccr.at[chemblid,0] #chemblid ( or cid) which is the right-most column in ccrfile
            except:
                ccr = chemblid # if the chemblid does not exist in ccrfile, the molecule has no scaffold shared and the chemblid is inserted in the variable 'ccr'
            ccr_list.append(ccr)
        return ccr_list

    def apply_ccr(self,ccrfile,smiles_list,activity_list,chembl_list):
        df_ccr = pd.read_csv(ccrfile,delimiter='|',header=None,index_col=2)
        new_smiles_list = []
        new_activity_list = []
        new_chembl_list = []
        ccr_list = []
        i=0
        j=0
        for smiles, activity,chembl in zip(smiles_list,activity_list,chembl_list):
            i+=1
            try:
                ccr = df_ccr.at[chembl,0] 
            except:
                ccr = smiles
            ccr_list.append(ccr)
            new_smiles_list.append(smiles)
            new_activity_list.append(activity)
            new_chembl_list.append(chembl)
 
        return new_smiles_list,new_activity_list,new_chembl_list,ccr_list
    

    def save_lists(self,filepath,chembl_list1,smiles_list1, activity_list1, ccr_list1):
        nmol = len(chembl_list1)
        if nmol == len(smiles_list1) and nmol == len(activity_list1):
             print("OK: The three lists have the same number of elements.")
        else:
             print("Error: The three lists must have the same number of elements.")
        f = open(filepath,'w') 
        f.write('chembl_id,smiles,ccr_scaffold,activity\n')
        for imol in range(nmol):
            f.write(chembl_list1[imol].replace(',','_'))
            f.write(',')
            f.write(smiles_list1[imol])
            f.write(',')
            f.write(ccr_list1[imol])
            f.write(',')
            f.write(str(activity_list1[imol]))
            f.write('\n')
        
      
    
