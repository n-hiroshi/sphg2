"""
Utility function of screening molecules with molecular weight
"""

import os,sys
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt


def screen(smiles_list,activity_list,chembl_list,activity_threshold=-1):
    minwt = 200.0
    maxwt = 600.0
    molwt_list=[]
    smiles_list1 = []
    activity_list1 =[]
    chembl_list1 = []
    for imol,(smiles,activity) in enumerate(zip(smiles_list,activity_list)):
        mol=Chem.MolFromSmiles(smiles)
        molwt = MolWt(mol)
        if minwt <= molwt and molwt <= maxwt and activity >= activity_threshold:
            molwt_list.append(molwt)
            smiles_list1.append(smiles)
            activity_list1.append(activity_list[imol])
            chembl_list1.append(chembl_list[imol])
    return smiles_list1, activity_list1, chembl_list1
