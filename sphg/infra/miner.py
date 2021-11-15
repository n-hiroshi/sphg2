# -*- coding: utf-8 -*-
import os,sys
import pickle
import numpy as np
from numba import jit
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import time

# miner
class miner():
    """
    main class used in STEP2 (list up all SPhGs from train-pos dataset and examine which mols include which SPhGs)
    """
    def __init__(self,AllNodeSPhGs,AllEdgeSPhGs,allocSPhG=10000,nCF=6):
        self.nCF=nCF
        allocSPhGvec = [0,0,5000,100000,500000,3000000,5000000]
        self.allocSPhG=allocSPhGvec[nCF]
        self.AllNodeSPhGs = AllNodeSPhGs
        self.AllEdgeSPhGs = AllEdgeSPhGs
        self.nmol = len(self.AllNodeSPhGs)

        # to avoid rate error pattern 
        for imol in range(self.nmol):
           if len(self.AllEdgeSPhGs[imol]) > 0:
               nnode = len(self.AllNodeSPhGs[imol][0,:])
               nedge = len(self.AllEdgeSPhGs[imol][0,:])
               break

        self.nnode = nnode
        self.nedge = nedge
        self.MineNodeSPhGs=np.zeros((self.allocSPhG,self.nnode),np.int32)
        self.MineEdgeSPhGs=np.zeros((self.allocSPhG,self.nedge),np.int32)
        self.MineHitvecs=np.zeros((self.allocSPhG,self.nmol),bool)

    def mine(self):
        kMineSPhG=0

        ## (1) Phamacophre mining
        print("################################")
        print("Pharmacophore mining start.")
        for imol in range(self.nmol):
            nodeSPhGs = self.AllNodeSPhGs[imol]
            edgeSPhGs = self.AllEdgeSPhGs[imol]
            
            #sys.stdout.write("%d "%len(SSPhGvecids)) 

            kMineSPhG,self.MineNodeSPhGs,self.MineEdgeSPhGs,self.MineHitvecs = \
            numbafunc1(nodeSPhGs,edgeSPhGs, \
                      self.MineNodeSPhGs,self.MineEdgeSPhGs,self.MineHitvecs, \
                      kMineSPhG,imol,self.nnode,self.nedge)
            
            #sys.stdout.write("\r"+str(imol+1)+"/"+str(self.nmol)+" kMineSPhG: "+str(kMineSPhG)+" allocSPhG: "+str(self.allocSPhG))
            #sys.stdout.flush()
            sys.stdout.write(str(imol+1)+"/"+str(self.nmol)+" kMineSPhG: "+str(kMineSPhG)+" allocSPhG: "+str(self.allocSPhG)+"\n")
            sys.stdout.flush()
        
        self.coveragevec = np.sum(self.MineHitvecs,axis=1)#/self.nmol
        print(self.coveragevec)
        print(len(self.coveragevec[self.coveragevec>0]))
        print(np.amax(self.coveragevec))
        print("Pharmacophore mining succeeded.")
        print("Make sure \"allocSPhG > kMineSPhG\"")
        print("################################")
              
        ## (2) Release redundant memory spaces for MineNode/EdgeSPhGs and MineHitvecs
        # (2-1) cut unesed spaces
        self.MineNodeSPhGs=self.MineNodeSPhGs[0:kMineSPhG]
        self.MineEdgeSPhGs=self.MineEdgeSPhGs[0:kMineSPhG]
        self.MineHitvecs=self.MineHitvecs[0:kMineSPhG,:]

        # (2-2) delete rare patterns of which the number is less than np.amin([10,self.nmol*0.05])
        self.coveragevec = np.sum(self.MineHitvecs,axis=1)
        #self.over_min_coveragevec = self.coveragevec > np.amin([10,self.nmol*0.05])
        if self.nCF>=4: 
            self.over_min_coveragevec = self.coveragevec > np.amin([10,self.nmol*0.05])
        else:
            self.over_min_coveragevec = self.coveragevec >= 0
        self.over_min_coveragevec = self.coveragevec >= 0

        self.MineNodeSPhGs=self.MineNodeSPhGs[self.over_min_coveragevec]
        self.MineEdgeSPhGs=self.MineEdgeSPhGs[self.over_min_coveragevec]
        self.MineHitvecs=self.MineHitvecs[self.over_min_coveragevec]
        self.coveragevec = np.sum(self.MineHitvecs,axis=1)
        #print("nSPhG over min_coveragevec=10: %d"%len(self.coveragevec))

        ## (3) Sort by coverage
        ordervec = np.argsort(self.coveragevec)[::-1]
        self.MineNodeSPhGs=self.MineNodeSPhGs[ordervec]
        self.MineEdgeSPhGs=self.MineEdgeSPhGs[ordervec]
        self.MineHitvecs=self.MineHitvecs[ordervec,:]

        return [self.MineNodeSPhGs,self.MineEdgeSPhGs,self.MineHitvecs]


@jit(nopython=True)
def numbafunc1(nodeSPhGs,edgeSPhGs,MineNodeSPhGs,MineEdgeSPhGs,MineHitvecs,kMineSPhG,imol,nnode,nedge):
    ##########################################################
    for iSSPhG, nodeSPhG in enumerate(nodeSPhGs):
        flagHit=False

        # (1) check if node/edgeSPhG have already filed in MineNode/EdgeSPhGs
        for iMineSPhG, MineNodeSPhG in enumerate(MineNodeSPhGs[0:kMineSPhG]):
            #if nodeSPhG == MineNodeSPhG:
            #    flagHit=True # temporalily set true
            flagSame=True
            for node_each,Mine_node_each in zip(nodeSPhG,MineNodeSPhG):
                if node_each != Mine_node_each: flagSame=False
            if flagSame: flagHit =True
                
            if flagHit:
                for iedge in range(nedge):
                    if MineEdgeSPhGs[iMineSPhG,iedge] != edgeSPhGs[iSSPhG,iedge]:
                         flagHit=False
                         break;
                if flagHit:
                    MineHitvecs[iMineSPhG,imol] = True
            if flagHit: break;
        #print(flagHit)

        # (2) file new node/edgeSPhG in MineNode/EdgeSPhGs
        if flagHit == False:
            MineNodeSPhGs[kMineSPhG] = nodeSPhG
            MineEdgeSPhGs[kMineSPhG,:] = edgeSPhGs[iSSPhG,:]
            MineHitvecs[kMineSPhG,imol] = True
            kMineSPhG+=1

    return kMineSPhG,MineNodeSPhGs,MineEdgeSPhGs,MineHitvecs
    ##########################################################

