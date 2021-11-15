import pytest
import os,sys
import numpy as np
from copy import deepcopy

class select():
    """
    Select important SPhGs with each criterion
    """
    def __init__(self,MineNodeSPhGs,MineEdgeSPhGs): 
        self.MineNodeSPhGs = MineNodeSPhGs
        self.MineEdgeSPhGs = MineEdgeSPhGs

    def makeSelectedEdgeNodeSPhGs(self,priority,nSPhG_toselect):
        tempMineNodeSPhGs = deepcopy(self.MineNodeSPhGs)#self.MineNodeSPhGs.deepcopy()
        tempMineEdgeSPhGs = deepcopy(self.MineEdgeSPhGs)#.deepcopy()

        priority_order = np.argsort(priority)[::-1]
        tempMineNodeSPhGs = tempMineNodeSPhGs[priority_order,:]
        tempMineEdgeSPhGs = tempMineEdgeSPhGs[priority_order,:]
        priority_sort = priority[priority_order]

        return tempMineNodeSPhGs[0:nSPhG_toselect,:], tempMineEdgeSPhGs[0:nSPhG_toselect,:]
             

    def selectCov(self,MineHitvecs,nSPhG_toselect):
        """
        Select important SPhGs with Converage criterion
        """
        priority = np.sum(MineHitvecs,axis=1) 
        return self.makeSelectedEdgeNodeSPhGs(priority,nSPhG_toselect)

    def selectScf(self,MineHitvecs,scaffold_list_pos,nSPhG_toselect):
        """
        Select important SPhGs with NScaffold criterion
        """
        nSPhG = MineHitvecs.shape[0] # MineHitvecs is nSPhG x nmol matrix, where nmol is num of train_pos.
        priority = np.zeros(nSPhG,int)
        for iSPhG in range(nSPhG):
            priority[iSPhG] = len(list(set(scaffold_list_pos[MineHitvecs[iSPhG,:]]))) 
            #len(list(set(...))) gets num of unique scaffolds
        return self.makeSelectedEdgeNodeSPhGs(priority,nSPhG_toselect)

    def selectGR(self,MineHitvecs,Search_Hitvecs_train_neg,nSPhG_toselect):
        """
        Select important SPhGs with Growth-rate criterion
        """
        nSPhG = MineHitvecs.shape[0] # MineHitvecs is nSPhG x nmol matrix, where nmol is num of train_pos.
        priority = np.zeros(nSPhG,int)
        cov_pos = np.sum(MineHitvecs,axis=1) 
        cov_neg = np.sum(Search_Hitvecs_train_neg,axis=1) 
        for iSPhG in range(nSPhG):
            assert len(cov_pos) > 0
            priority[iSPhG] = cov_pos[iSPhG]/(cov_neg[iSPhG]+0.00001) # avoid divide-by-zero

        return self.makeSelectedEdgeNodeSPhGs(priority,nSPhG_toselect)


