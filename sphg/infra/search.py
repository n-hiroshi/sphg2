import os,sys
import pickle
import numpy as np
from numba import jit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time

# miner
class search():
    """
    Search which mols his which sphg. (accelerated by numba)
    """
    def __init__(self,MineNodeSPhGs,MineEdgeSPhGs,allocSPhG=10000):
        nCF=6
        self.MineNodeSPhGs=MineNodeSPhGs
        self.MineEdgeSPhGs=MineEdgeSPhGs
        self.nMineSPhGs= len(MineNodeSPhGs)
        self.nnode= len(MineNodeSPhGs[0,:])
        self.nedge= len(MineEdgeSPhGs[0,:])
        assert self.nMineSPhGs == len(MineEdgeSPhGs)
        

    def search(self,AllNodeSPhGs,AllEdgeSPhGs):
        """
        Main function of Search class
        """

        self.nmol = len(AllNodeSPhGs)
        assert self.nmol == len(AllEdgeSPhGs)

        ## (1) define Hitvecs to return
        self.SearchHitvecs = np.zeros((self.nMineSPhGs,self.nmol),bool)

        ## (2) Phamacophre search
        #print("# Pharmacophore search")
        for imol in range(self.nmol):
            nodeSPhGs = AllNodeSPhGs[imol]
            edgeSPhGs = AllEdgeSPhGs[imol]
            
            #print(nodeSPhGs)
            #print(edgeSPhGs)

            # accumulate a search_resulte of ach mols in each of the columns of self.SearchHitvecs
            self.SearchHitvecs = \
            numbafunc2(imol,self.nnode,self.nedge,nodeSPhGs,edgeSPhGs,self.MineNodeSPhGs,self.MineEdgeSPhGs,self.SearchHitvecs)
            
            #sys.stdout.write("\r"+str(imol+1)+"/"+str(self.nmol))
            #sys.stdout.flush()
        
        #self.coveragevec = np.sum(self.MineHitvecs,axis=1)#/self.nmol
        #print(self.coveragevec)
        #print(len(self.coveragevec[self.coveragevec>0]))
        #print(np.amax(self.coveragevec))
        #print("\nPharmacophore search succeeded.")
        #print("################################")
              
        return self.SearchHitvecs

@jit(nopython=True)
def numbafunc2(imol,nnode,nedge,nodeSPhGs,edgeSPhGs,MineNodeSPhGs,MineEdgeSPhGs,SearchHitvecs):
    ##########################################################
    for iSPhG, nodeSPhG in enumerate(nodeSPhGs):
        flagHit=False

        for iMineSPhG, MineNodeSPhG in enumerate(MineNodeSPhGs):
            #if nodeSPhG == MineNodeSPhG:
            #    flagHit=True # temporalily set true
            flagSame=True
            for node_each,Mine_node_each in zip(nodeSPhG,MineNodeSPhG):
                if node_each != Mine_node_each: flagSame=False
            if flagSame: flagHit =True

            if flagHit:
                for iedge in range(nedge):
                    if MineEdgeSPhGs[iMineSPhG,iedge] != edgeSPhGs[iSPhG,iedge]:
                         flagHit=False
                         break;
                if flagHit:
                    SearchHitvecs[iMineSPhG,imol] = True
            if flagHit: break;

    return SearchHitvecs
    ##########################################################
