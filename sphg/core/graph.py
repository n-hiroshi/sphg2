import os,sys
import numpy as np
from numba import njit,jit,i1,i8,b1
import copy
import itertools

class sphcg():
    def __init__(self,distmat,atomListForFeats,featsListMol,featsListForAtom):
        self.distmat=distmat
        self.atomListForFeats=atomListForFeats
        self.featsListMol=featsListMol
        self.featsListForAtom=featsListForAtom
    def print(self):
        print('atomListForFeats')
        print(self.atomListForFeats)
        print('featsListMol')
        print(self.featsListMol)
        print('featsListForAtom')
        print(self.featsListForAtom)
        print('distmat')
        print(self.distmat)

class Graph(): 
    """
    make Mol-SPhG and SPhG corresponding to Fig 3(a) and (b)
    """
  
    def __init__(self, vertices): 
        self.V = np.int8(vertices) #num of nodes 
        self.graphnp = np.zeros((self.V,self.V),dtype=np.int8)

    def isTree(self,src=0):
        self.graphnp = np.int8(self.graphnp)
        return nb_isTree(self.V,self.graphnp)

    def del_m1_and_convertlist(self,ndarray0,dim):
        if dim == 1:
            list0 = [v for v in ndarray0 if v >=0]   
        elif dim == 2:
            list0 = [[v for v in ndarray1 if v >=0]  for ndarray1 in ndarray0] 
        elif dim == 3: # only for paths
            list0 = [[[v for v in ndarray1 if v >=0]  for ndarray1 in ndarray2] for ndarray2 in ndarray0] 
            list0 = [[[v for v in list1 if v >=0]  for list1 in list2 if len(list1)>0] for list2 in list0] 
        return list0

    def dijkstra(self, src):   
        return nb_dijkstra(self.V,self.graphnp,src)

    def shortestPaths(self,src):
        dist,paths = nb_shortestPaths(self.V,self.graphnp,src)
        return dist,paths
    def makeFeatsListForAtom(self,subAtomListForFeats,subFeatsListMol,subgraph):
        nnodes=subgraph.shape[0]
        subFeatsListForAtom = [[] for i in range(nnodes)]
        assert len(subAtomListForFeats) == len(subFeatsListMol)
        for subFeatsAtomList, subFeatsFamily in zip(subAtomListForFeats,subFeatsListMol):
            for iatom in subFeatsAtomList:
                 if subFeatsFamily == 8:
                     if len(subFeatsListForAtom[iatom])==0: subFeatsListForAtom[iatom]=[8]
                 elif subFeatsListForAtom[iatom]== [8]:
                     subFeatsListForAtom[iatom]=[subFeatsFamily]
                 else:
                     subFeatsListForAtom[iatom].append(subFeatsFamily)
        subFeatsListForAtom = [sorted(familyList) for familyList in subFeatsListForAtom]
        return subFeatsListForAtom

    def abstractGraph(self,atomListForFeats, featsListMol,verbose=False):
        """
        make Mol-SPhGs from molecular structures and features assignments corresponding to Fig. 3(a)
        """
        # (1) Assign juncitons
        #     Detect the junctions, which are the nodes with more than 3 edges connected 
        #     and Add them to atomListForFeats, featsFamilyList
        #     The junctinos are labeld as '9' in featsListMol
        junctionTF = np.sum(self.graphnp,axis=1)>=3        
        for iatom, TF in enumerate(junctionTF):
            if TF:
                atomListForFeats.append([iatom])
                featsListMol.append(9)
        assert len(atomListForFeats) == len(featsListMol)

        if verbose==True: print('junctionTF (1)')
        if verbose==True: print(junctionTF)
        featsListForAtom=self.makeFeatsListForAtom(atomListForFeats,featsListMol,self.graphnp)
        distmat=self.graphnp
        sphcg1=sphcg(distmat,atomListForFeats,featsListMol,featsListForAtom)
        if verbose==True: print('sphcg (1)')
        if verbose==True: sphcg1.print()
        
        # (2) Reduce graph
        #     Delete the nodes with one or two edges connected and no CF assignment
        sphcg1=self.__shrinkGraph(sphcg1)
        sphcg1=self.__cleanGraph(sphcg1)

        if verbose==True: print('sphcg (2)')
        if verbose==True: sphcg1.print()

        self.sphcgList=[sphcg1]
        return self.sphcgList


    def abstractSubGraph(self,combiFeats,convertAromaticToBond=True,verbose=False):
        """
        make SPhGs from Mol-SPhGs and the selected features corresponding to Fig. 3(b)
        """
        subSphcgList = copy.deepcopy(self.sphcgList)
        for sphcg0 in subSphcgList:
            # (1) Delete all CFs except ones in combiFeats and junction [9] 
            featsListMol_temp = []
            atomListForFeats_temp =  [] 
            for ifeats,featsFamily in enumerate(sphcg0.featsListMol):
                if ifeats in combiFeats:
                    featsListMol_temp.append(sphcg0.featsListMol[ifeats])
                    atomListForFeats_temp.append(sphcg0.atomListForFeats[ifeats])  
            featsAtomSet = list(set([v for featsAtomList in atomListForFeats_temp for v in featsAtomList]))
            for iatom in range(sphcg0.distmat.shape[0]):
                if not(iatom in featsAtomSet):
                    featsListMol_temp.append(9)
                    atomListForFeats_temp.append([iatom])
            sphcg0.featsListMol = featsListMol_temp
            sphcg0.atomListForFeats = atomListForFeats_temp
            sphcg0.featsListForAtom=self.makeFeatsListForAtom(sphcg0.atomListForFeats,sphcg0.featsListMol,sphcg0.distmat)
            if verbose: print('abstractSubGraph (1)')
            if verbose: sphcg0.print()
            # (2) Delete the nodes with one or two edges connected and no CF assignment
            sphcg0=self.__shrinkGraph(sphcg0)
            sphcg0=self.__cleanGraph(sphcg0)
            sphcg0.featsListForAtom=self.makeFeatsListForAtom(sphcg0.atomListForFeats,sphcg0.featsListMol,sphcg0.distmat)
            if verbose: print('abstractSubGraph (2)')
            if verbose: sphcg0.print()
        
        # (3) Delete more-than-and-equal-to-three-edges nodes with feats = 6 or 9 repeatedly
        #     an store all patterns
        isChange = True

        # This while-loop is cooresponding to the loop on the panelis in Fig.3(b)
        while isChange==True:
            isChange=False
            subSphcgList_new = []
            for sphcg1 in subSphcgList:
                isSphcgUpdate=False
                for atom0, featsList in enumerate(sphcg1.featsListForAtom):
                    # (3-1) Delete more-than-three-edges nodes with feats = 6(=aromatic) or 9(=junctinos)
                    if (featsList == [9] or featsList == [6] or featsList == [6,6] or featsList == [6,6,6]) and np.sum(sphcg1.distmat[atom0,:]>0)>=3:
                        sphcg2=copy.deepcopy(sphcg1)
                        sphcg2.distmat[atom0,:]=0
                        sphcg2.distmat[:,atom0]=0
                        if atom0 == 0:
                            dist, prev = nb_dijkstra(sphcg2.distmat.shape[0],sphcg2.distmat,1)
                        else:
                            dist, prev = nb_dijkstra(sphcg2.distmat.shape[0],sphcg2.distmat,0)
                            
                        # 'dist=127' means no-noconnection. 
                        # Skip the sphcg if an independent node exist other than the cut ndode.
                        # and reduce graph
                        if np.sum(dist==127)==1:
                            isChange=True
                            isSphcgUpdate=True
                            sphcg3=self.__shrinkGraph(sphcg2)
                            sphcg3=self.__cleanGraph(sphcg3)
                            # Confirm that all CFs except junctions junction remains
                            assert [v for v in sphcg3.featsListMol if v!=9] == [v for v in sphcg2.featsListMol if v!=9]
                            subSphcgList_new.append(sphcg3)
                if isSphcgUpdate==False: subSphcgList_new.append(sphcg1)
            if isChange: subSphcgList = copy.deepcopy(subSphcgList_new)    

        # (4) delete long bonds in cyclic structure (e.g. macrocyclic)
        for sphcg0 in subSphcgList:
            isChange_atLeast1=True
            while isChange_atLeast1:
                V=sphcg0.distmat.shape[0]
                isChange_atLeast1=False
                
                for src in range(V):
                    #if not(src in aromList):
                    sphcg0.distmat,isChange = nb_brakeMacrocyclic(V,sphcg0.distmat,src)
                    if isChange: isChange_atLeast1=True
                if isChange_atLeast1:
                    sphcg0=self.__shrinkGraph(sphcg0)
                    sphcg0=self.__cleanGraph(sphcg0)
            if verbose==True: print('\n# abstractSubGraph (3)----')
            if verbose==True: sphcg0.print()

        # (5) Keeps the minimum sphcgs in the all generated sphcgs.
        minallpaths = sys.maxsize
        subSphcgList_old = copy.deepcopy(subSphcgList)
        for sphcg0 in subSphcgList_old:
            kallpaths = np.sum(np.sum(sphcg0.distmat))
            if kallpaths < minallpaths:
                minallpaths=kallpaths
                subSphcgList=[sphcg0]
            elif kallpaths == minallpaths:
                subSphcgList.append(sphcg0)

        # (6) Reduce multiple sphcgs to one
        subSphcgList0=copy.deepcopy(subSphcgList)
        subSphcgList1=[]
        for sphcg0 in subSphcgList0:
            isSkip=False
            for sphcg1 in subSphcgList1:
                isEqual=True
                if sphcg0.distmat.shape[0] == sphcg1.distmat.shape[0] and sphcg0.distmat.shape[1] == sphcg1.distmat.shape[1]:
                    if np.sum(np.sum(np.abs(sphcg0.distmat - sphcg1.distmat)))>0: isEqual=False
                else: isEqual=False
                if sphcg0.atomListForFeats != sphcg1.atomListForFeats: isEqual=False
                if sphcg0.featsListMol != sphcg1.featsListMol: isEqual=False
                if sphcg0.featsListForAtom != sphcg1.featsListForAtom: isEqual=False
                if isEqual==True: isSkip=True 
            if isSkip==False: subSphcgList1.append(sphcg0) 
            
        self.subSphcgList = subSphcgList1
        # (7) convert aromatic node features to aromatic edge features
        if convertAromaticToBond: subSphcgList=self.convertAromaticToBond(subSphcgList) 
        if verbose==True: print('\n# abstractSubGraph (5)----')
        if verbose==True: sphcg0.print()

        return subSphcgList
        

    def __shrinkGraph(self,sphcg0):
        # (1) Reduce the 1-edge node and 2-edges node without any CF assignment
        #     Reduce junctions with 1 edge or 2 edges connected
        #     Repeat above until convergence. 
        #     For example, the junctions with 1 edge or 2 edges generated during the loop.
        #     In this function 'node' and 'atom' are used as the same meaning because atoms
        #     in molecules are graduaally erased and the remaining atoms are called nodes in SPhG or MolSPhG.
        distmat=sphcg0.distmat
        atomListForFeats=sphcg0.atomListForFeats
        featsListMol=sphcg0.featsListMol
        featsListForAtom=sphcg0.featsListForAtom

        natom=len(featsListForAtom)
        isChange = True

        # This while-loop is cooresponding to the loop on the reduce-graph panel in Fig.3(a)
        while isChange==True:
            isChange = False
            for atom0, featsList in enumerate(featsListForAtom):
                # (1-1) junction nodes and no-CF nodesCF
                if featsList == [] or featsList == [9]:
                     # (1-1-1) one-edge-connected nodes 
                     if np.sum(distmat[atom0,:]>0)==1: 
                         isChange=True
                         distmat[atom0,:]=0
                         distmat[:,atom0]=0
                         atomListForFeats = [[atom2 for atom2 in featsAtomList if atom2!=atom0] for featsAtomList in atomListForFeats]
                     # (1-1-2) two-edges-conneced nodes
                     elif np.sum(distmat[atom0,:]>0)==2: 
                         isChange=True
                         adj0=-99
                         for atom1 in range(natom):
                             if adj0==-99  and distmat[atom0,atom1]>0: adj0=atom1
                             elif adj0>=0  and distmat[atom0,atom1]>0: adj1=atom1
                         dist = distmat[atom0,adj0]+distmat[atom0,adj1]
                         if distmat[adj0,adj1]==0 or distmat[adj0,adj1]>dist: 
                             distmat[adj0,adj1]=dist
                             distmat[adj1,adj0]=dist
                         distmat[atom0,:]=0
                         distmat[:,atom0]=0
                         atomListForFeats = [[atom2 for atom2 in featsAtomList if atom2!=atom0] for featsAtomList in atomListForFeats]
                # (1-2) nodes in the aromatic group
                elif featsList == [6] or featsList == [6,9] or featsList == [8] or featsList == [8,9]:
                    #(1-2-1) detect the nodes in aromatic group connecting to the outside of the aromatic group
                    for featsFamily, featsAtomList in zip(featsListMol,atomListForFeats): 
                        if atom0 in featsAtomList and (featsFamily==6 or featsFamily==8): 
                            ownAtomList=featsAtomList

                    #algo1
                    isKeep=False
                    for atom1,d in enumerate(distmat[atom0,:]):
                        if d > 0: # retrieve atoms ( called atom1 ) connecting atom0 
                            # Keep atom0 if at leaset one atom outside of the aromatic group is connecting to atom0
                            if not(atom1 in ownAtomList): isKeep=True 

                    # (1-2-2) reduce the atom judged in (1-2-1)
                    if isKeep == False:
                         # (1-2-2-1) atoms with one connection
                         if np.sum(distmat[atom0,:]>0)==1: 
                             isChange=True
                             distmat[atom0,:]=0
                             distmat[:,atom0]=0
                             atomListForFeats = [[atom2 for atom2 in featsAtomList if atom2!=atom0] for featsAtomList in atomListForFeats]
                         # (1-2-2-2) atoms with two connection
                         elif np.sum(distmat[atom0,:]>0)==2: 
                             isChange=True
                             adj0=-99
                             for atom1 in range(natom):
                                 if adj0==-99  and distmat[atom0,atom1]>0: adj0=atom1
                                 elif adj0>=0  and distmat[atom0,atom1]>0: adj1=atom1
                             dist = distmat[atom0,adj0]+distmat[atom0,adj1]
                             if distmat[adj0,adj1]==0 or distmat[adj0,adj1]>dist: 
                                 distmat[adj0,adj1]=dist
                                 distmat[adj1,adj0]=dist
                             distmat[atom0,:]=0
                             distmat[:,atom0]=0
                             atomListForFeats = [[atom2 for atom2 in featsAtomList if atom2!=atom0] for featsAtomList in atomListForFeats]

        assert len(featsListForAtom) == distmat.shape[0]
        assert len(atomListForFeats) == len(featsListMol)
        sphcg0.distmat=distmat
        sphcg0.atomListForFeats=atomListForFeats
        sphcg0.featsListMol=featsListMol
        featsListForAtom=self.makeFeatsListForAtom(atomListForFeats,featsListMol,distmat)
        sphcg0.featsListForAtom=featsListForAtom
        return sphcg0

    def __cleanGraph(self,sphcg0):
        # This function cleans sphcg0 (distmat, atomListForFeats, and featsListForAtom)
        # (1) reduce the rows and cols of isolated atoms
        distmat=sphcg0.distmat
        atomListForFeats=sphcg0.atomListForFeats
        featsListMol=sphcg0.featsListMol
        featsListForAtom=sphcg0.featsListForAtom
        keepatomTF = np.sum(distmat>0,axis=0)>=1
        distmat=distmat[keepatomTF,:]
        distmat=distmat[:,keepatomTF]

        # (2) delete the indexes of the atoms from atomListForFeats
        atomListForFeats = [[atom2 for atom2 in featsAtomList if keepatomTF[atom2] ] for featsAtomList in atomListForFeats]
        # (3) Delete lists with no atoms in featsFamily and featsAtimListList(=only junction is deleted)
        for featsAtomList, featsFamily in zip(atomListForFeats,featsListMol):
            featsListMol = [featsFamily for ifeats, featsFamily in enumerate(featsListMol) if len(atomListForFeats[ifeats]) > 0] 
            atomListForFeats = [featsAtomList for featsAtomList in atomListForFeats if len(featsAtomList) > 0] 

        # (4) Atom indexes used in featsAtimListList are the indexes of the sphcg before reduction process
        atomListForFeats = [[np.sum(keepatomTF[0:atom0]) for atom0 in  featsAtomList] for featsAtomList in atomListForFeats] 

        # (5) Clean junction feature assignments
        featsListForAtom=self.makeFeatsListForAtom(atomListForFeats,featsListMol,distmat)
        featsListForAtom_temp = []
        for atom0, familyList in enumerate(featsListForAtom):
            # atoms only with the junction featrue remains
            if familyList==[9]: 
                featsListForAtom_temp.append(familyList)
            # junction fetures are deleted if an atoms has the junction and ohter features.
            elif len(familyList)>=2 and 9 in familyList:  
                familyList_temp = [feats for feats in familyList if feats != 9]
                featsListForAtom_temp.append(familyList_temp)
                for ifeats, (featsAtomList,featsFamily) in enumerate(zip(atomListForFeats,featsListMol)):
                    if featsFamily == 9 and featsAtomList == [atom0]: featsToPop=ifeats
                featsListMol.pop(featsToPop)
                atomListForFeats.pop(featsToPop)
                #print(featsListForAtom_temp)
            else:
                featsListForAtom_temp.append(familyList)
        featsListForAtom = featsListForAtom_temp


        assert len(featsListForAtom) == distmat.shape[0]
        assert len(atomListForFeats) == len(featsListMol)
        sphcg0.distmat=distmat
        sphcg0.atomListForFeats=atomListForFeats
        sphcg0.featsListMol=featsListMol
        featsListForAtom=self.makeFeatsListForAtom(atomListForFeats,featsListMol,distmat)
        sphcg0.featsListForAtom=featsListForAtom
        return sphcg0

    def convertAromaticToBond(self,sphcgList): 
        # This function convert aromatic node featrue to aromatic bond featrue
        addDistForAromaticBonds=30
        for sphcg0 in sphcgList:
            # (1) 30 is added to bond-length(=dist) for aromatic bond
            for featsFamily, featsAtomList in zip(sphcg0.featsListMol,sphcg0.atomListForFeats):
                if featsFamily == 6 or featsFamily == 8:
                    for combi in itertools.combinations(featsAtomList,2):
                        # the bond shared with two aromatic rings
                        if sphcg0.distmat[combi[0],combi[1]] >0 and sphcg0.distmat[combi[0],combi[1]] <30:  
                            sphcg0.distmat[combi[0],combi[1]]+=addDistForAromaticBonds
                            sphcg0.distmat[combi[1],combi[0]]+=addDistForAromaticBonds               

            # (2) change the feature index of atoms to 8 if the aromatic bond connected to the atoms
            #     If the atoms with an aromatic ring feature and no other atoms shares the same ring feature,
            #     the featrue index of the nodes remain '6' and no aromatic BOND remained.
            for ifeats, featsAtomList in enumerate(sphcg0.atomListForFeats):
                if sphcg0.featsListMol[ifeats] == 6 and len(featsAtomList) >=2:
                     #print('feats 6 to 9 (5)')
                     sphcg0.featsListMol[ifeats]=8
                     
            sphcg0.featsListForAtom=self.makeFeatsListForAtom(sphcg0.atomListForFeats,sphcg0.featsListMol,sphcg0.distmat)
        return sphcgList

@njit
def nb_dijkstra(V,graphnp,src):   
    dist = 127*np.ones(V,dtype=np.int8)
    prev = -1*np.ones((V,V),dtype=np.int8)
    dist[src] = 0
    sptSet = np.zeros(V,dtype=np.bool_)#[False] * V   
    for cout in range(V): 
        ##################
        ## minDistance ###
        min = sys.maxsize
        for v in range(V): 
            if dist[v] < min and sptSet[v] == False: 
                min = dist[v] 
                min_index = v   
        ##################
        u = min_index
        sptSet[u] = True 
        #sptSet[u] = True  
        for v in range(V):
            if graphnp[u][v] > 0 and sptSet[v] == False: 
                if dist[v] > dist[u] + graphnp[u][v]: 
                    dist[v] = dist[u] + graphnp[u][v] 
                    prev[v,0] = u
                    prev[v,1:] = -1
                elif dist[v] == dist[u] + graphnp[u][v]: # more than one minimum length paths
                    pos = np.sum(prev[v,:]>=0)
                    prev[v,pos] = u
                #elif dist[v] < dist[u] + graphnp[u][v]: # ignore non-minimum paths
    return dist,prev
@njit
def nb_brakeMacrocyclic(V,graphnp,src):   
    isChange=False
    dist = 127*np.ones(V,dtype=np.int8)
    prev = -1*np.ones((V,V),dtype=np.int8)
    dist[src] = 0
    sptSet = np.zeros(V,dtype=np.bool_)#[False] * V   
    for cout in range(V): 
        #u = self.minDistance(dist, sptSet) 
        ##################
        ## minDistance ###
        min = sys.maxsize
        for v in range(V): 
            if dist[v] < min and sptSet[v] == False: 
                min = dist[v] 
                min_index = v   
        ##################
        u = min_index
        sptSet[u] = True 
        #sptSet[u] = True  
        for v in range(V):
            if graphnp[u][v] > 0 and sptSet[v] == False: 
                #if dist[v] > dist[u] + graphnp[u][v]: 
                if dist[v] >= dist[u] + graphnp[u][v]: 
                    # If the directed paths from src is longer than other paths, 
                    # the directed path is deleted.
                    # Here, the distances of the nodes connecting to src are registered in dist because of 
                    # the breadth first search
                    if dist[v]< 127 and prev[v,0]==src and u!=src:# and dist[v] + (dist[u]+graphnp[u][v]) >=7:
                        isChange=True
                        graphnp[src,v]=0
                        graphnp[v,src]=0
                    dist[v] = dist[u] + graphnp[u][v] 
                    prev[v,0] = u
                    prev[v,1:] = -1
                elif dist[v] == dist[u] + graphnp[u][v]: # more than one minimum length paths
                    pos = np.sum(prev[v,:]>=0)
                    prev[v,pos] = u
                #elif dist[v] < dist[u] + graphnp[u][v]: # ignore non-minimum paths
    return graphnp,isChange

@njit
def nb_shortestPaths(V,graphnp,src):
    dist = 127*np.ones(V,dtype=np.int8)
    mpath=16
    paths = -1*np.ones((V,mpath,V),dtype=np.int8) #max num of shortest-paths from src to another atom
    dist[src] = 0
    sptSet = np.zeros(V,dtype=np.bool_)#[False] * V   
    for cout in range(V): 

        ##################
        ## minDistance ###
        min = sys.maxsize
        for r in range(V): 
            if dist[r] < min and sptSet[r] == False: 
                min = dist[r] 
                min_index = r   
        ##################
        u = min_index
        sptSet[u] = True 
        # u: an edge just searched, v: a next frontier to be searched 
        for v in range(V):
            if graphnp[u][v] > 0 and sptSet[v] == False: 
                if dist[v] > dist[u] + graphnp[u][v]: 
                    dist[v] = dist[u] + graphnp[u][v] 

                    if u == src:
                        paths[v,0,0]=src
                    else:
                        paths[v,:,:]=paths[u,:,:]
                        # add u to all paths from src to u and register all of them as the paths to v
                        for j in range(mpath):
                            if np.sum(paths[v,j,:]>=0)>0:#loop of each path from src to u
                                pos = np.sum(paths[v,j,:]>=0)
                                paths[v,j,pos] = u

                elif dist[v] == dist[u] + graphnp[u][v]: 
                    for i in range(mpath):
                        if np.sum(paths[u,i,:]>=0)>0:
                            for j in range(mpath):
                                if np.sum(paths[v,j,:]>=0)==0:
                                    paths[v,j,:]=paths[u,i,:]
                                    pos = np.sum(paths[v,j,:]>=0)
                                    paths[v,j,pos] = u
                                    break

    # add the end points to paths
    for i in range(V):
        for j in range(mpath):
            if np.sum(paths[i,j,:]>=0)>0:
                pos = np.sum(paths[i,j,:]>=0)
                paths[i,j,pos] = i
    paths[src,0,0]=src
    return dist,paths

#@njit
@jit('b1(i1,i1[:,:])',nopython=True)
def nb_isTree(V,graphnp):   
    src=0
    flagTree = True
    seen  = np.zeros(V,dtype=np.bool_)#[False] * V   
    #todo  = np.zeros(V,dtype=np.bool_)#[False] * V   
    final = np.zeros(V,dtype=np.bool_)#[False] * V   
    seen[src] = True
    todo=[src]
    u = src
    maxite=100
    i=0
    # u: current position, v: next condiate
    while len(todo) >= 1 and i < maxite and flagTree: 
        i+=1
        u = todo.pop(-1)
        seen[u] = True
        numSeen=0
        for v in range(V):
            if graphnp[u][v] >= 1 and seen[v] == False: # go down to the next level
                todo.append(v)
            elif graphnp[u][v] >= 1 and seen[v] == True: # go down to the next level
                numSeen+=1
        if numSeen >= 2: flagTree=False

    return flagTree
