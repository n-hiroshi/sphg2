# -*- coding: utf-8 -*-
import os,sys
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D,DrawingOptions
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from itertools import combinations 
from itertools import permutations
from numba import jit,njit
from sphg.core.graph import Graph
from sphg.infra.canonicalize import Canonicalizer
from math import factorial

class xmol():
    """
    Main class for each molecules.
    The instances generates Mol-SPhG and SPhGs for the correspoinding molecule
    """
    def __init__(self,mol,molid=None):
        self.mol=mol
        self.smiles=Chem.MolToSmiles(self.mol)
        self.natom = mol.GetNumAtoms()
        if molid==None: self.molid='no mol name'
        else: self.molid=molid
        self.adjmat = GetAdjacencyMatrix(self.mol,useBO=False)    
        self.FeatFamilyDict = {"Donor":1,"Acceptor":2,"NegIonizable":3,"PosIonizable":4,"ZnBinder":5,"Aromatic":6,"Hydrophobe":7}
        self.FeatTypeDict = {"Donor":1,"Acceptor":2,"NegIonizable":3,"PosIonizable":4,"ZnBinder":5,"Aromatic":6,"Hydrophobe":7,"LumpedHydrophobe":8}

    #############################################################################################
    def getChemicalFeats(self,phg=False,fdefFile='BaseFeatures.fdef',verbose=False): 
        '''
        get Chemical Features for self.mol and set Graph details
        '''

        # (1)
        feats_nonHydrophobe = self.__getAllChemicalFeatures(fdefFile)
        # (2)
        self.featsFamilyList, self.featsAtomListGraph = self.__selectRequiredFeatures(feats_nonHydrophobe, phg=phg)
        # (3)
        self.__setGraphInfo()
  
    # (1) subfunciton in getChemicalFeats()
    def __getAllChemicalFeatures(self,fdefFile):
        featFact = ChemicalFeatures.BuildFeatureFactory(fdefFile)
        feats=featFact.GetFeaturesForMol(self.mol)
        
        feats_nonHydrophobe=[]
        feats_Hydrophobe=[]
        for feat in feats:#self.feats:
            featType=feat.GetType()
            featFamily=feat.GetFamily()
        
            if "Hydrophobe" in featFamily: 
                feats_Hydrophobe.append(feat)
            else: feats_nonHydrophobe.append(feat)
           
        aromatic_atom_list = []
        for ifeat0, feat0 in enumerate(feats_nonHydrophobe):
            if "Aromatic" in feat0.GetFamily():
                aromatic_atom_list+=list(feat0.GetAtomIds())
        
        return feats_nonHydrophobe # feats_Hydrophobe is never used in this program

    # (2) subfunciton in getChemicalFeats()
    def __selectRequiredFeatures(self,feats_nonHydrophobe, phg=False):
        nfeat = len(feats_nonHydrophobe)
        
        featsFamilyList = []
        featsAtomListGraph = []
        
        if phg:
        #NOTE: phg includes hydrophobic and zinc binder in the following paper
        # J Chem Inf Model. 2020,60,2073-2081 and J. Chem. Inf. Model. 2021, 61, 7, 3348–3360
           for ifeat0, feat0 in enumerate(feats_nonHydrophobe):
               if self.FeatFamilyDict[feat0.GetFamily()] in [1,2,3,4,6]:
                   featsFamilyList.append(self.FeatFamilyDict[feat0.GetFamily()])
                   featsAtomListGraph.append(list(feat0.GetAtomIds()))
           
        else: #sphg
           for ifeat0, feat0 in enumerate(feats_nonHydrophobe):
               if self.FeatFamilyDict[feat0.GetFamily()] in [1,2,3,4,6]:
                   featsFamilyList.append(self.FeatFamilyDict[feat0.GetFamily()])
                   featsAtomListGraph.append(list(feat0.GetAtomIds()))
        return featsFamilyList, featsAtomListGraph
            
        
    # (3) subfunciton in getChemicalFeats()
    def __setGraphInfo(self):
        self.adjmat = GetAdjacencyMatrix(self.mol,useBO=False)    
        self.g = Graph(self.mol.GetNumAtoms())
        self.g.graphnp=self.adjmat.astype(np.int8)

        # prepare distance matrix and all paths-list   
        self.distmatGraph=np.zeros((self.natom,self.natom),np.int8)
        self.pathsListGraph=[]
        for atom0 in range(self.natom):
            distvec, paths = self.g.shortestPaths(atom0)
            paths=self.g.del_m1_and_convertlist(paths,3)
            self.distmatGraph[atom0,:] = distvec
            self.pathsListGraph.append(paths)


    ##################################################################
    def encodeSPhG(self,distmat,featsListForAtom,nCF=6,verbose=False):
        '''
        convert (featsListForAtoam(nodes),distmat) to (nodesid and edgesid)
        '''
        if verbose: print("#encodeSPhG")
        if verbose: print(distmat)
        if verbose: print(featsListForAtom)

        #(1) check the sizes of distmat and featsListForAtom
        assert len(distmat) == len(featsListForAtom)

        #(2) convert featsListForAtom to nodes and distmat to edges
        edges = [[u,v,distmat[u,v]] for u,distrow in enumerate(distmat) for v, dist in enumerate(distrow) if dist!=0 and u<v]
        edges1 = edges.copy()
        nodes = featsListForAtom 

        #(3) canonicalization
        c = Canonicalizer()
        nodes, edges = c.canonicalize(nodes,edges)

        if verbose: print('nodes')
        if verbose: print(nodes)
        if verbose: print('edges')
        if verbose: print(edges)

        #(4) convert nodes(=FeatsListAtom) to nodesid
        nodesid = np.zeros(nCF*4,dtype=np.int32)
        for inode,node in enumerate(nodes):
            nodesid[inode] = node

        #(5) convert edges to edgesid
        edgesid = np.zeros(nCF*4,dtype=np.int32)
        for iedge,vec in enumerate(edges):
            if vec[2] == -1: vec[2]=0 # Convert dist=-1 (= inclusion) to 0
                                      # Convert dist=0 to undefined
            edgesid[iedge]=vec[0]*10000+vec[1]*100+vec[2]
        if verbose: print('nodesid and edgeid')
        if verbose: print(nodesid)
        if verbose: print(edgesid)
        for edgeid in edgesid: assert edgeid >= 0

        return nodesid,edgesid

    ##################################################################
    def decodeSPhG(self,nodeSPhG,edgeSPhG):
        '''
        convert (nodesid and egesid) to (distmat and nodes(=featsListForAtom))
        '''

        #(1) convert nodeSPhG to featsFamilySetList
        nnodes = np.sum(nodeSPhG!=0)
        nedges = np.sum(edgeSPhG!=0)
        nodeSPhG=nodeSPhG[0:nnodes]
        edgeSPhG=edgeSPhG[0:nedges]
        #print(nodeSPhG)
        #print(edgeSPhG)

        nodes = []
        for inode,nodeid in enumerate(nodeSPhG):
            node = []
            v=nodeid
            while(v!=0):
                node.append(v%10)
                v=v//10
                #print(v)
            node.sort()
            nodes.append(node)

        featsFamilySetList = nodes
        
        #(2) convert edgesid to distmat
        adjmat = np.zeros((nnodes,nnodes),int)
        for iedge in range(nedges):
            edgeHash = edgeSPhG[iedge]
            idx0 = int(edgeHash//10000)
            edgeHash1 = edgeHash%10000
            idx1 = int(edgeHash1//100)
            dist = int(edgeHash1%100)
            if dist==0 and idx0 != idx1: dist=-1
            adjmat[idx0,idx1] = dist
            adjmat[idx1,idx0] = dist

        #print(adjmat)
        distmat = adjmat
 
        return nodes,distmat

    ##################################################################
    def getAllSPhGs(self,nCF=6,phg=False,verbose=False,stat=0):
        '''
        calculate all (S)PhGs for a xmol. All combinations of 6 CFs out of all CFs in the Mol-SPhG,
        where 6 is a default value you can set any integer values >2.
        '''

        #(1) get all Chemical Features in the mol 
        self.getChemicalFeats(phg=phg,verbose=False)
        nfeats = len(self.featsFamilyList)
        j = self.nCr(nfeats,nCF)
        print(' molid: %s'%self.molid)
        print(' num of combinations: %d'%j)
        

        #(2) calc a Mol-SPhG
        #sphgList =  self.g.abstractGraph(self.featsAtomListGraph,self.featsFamilyList)
        self.g.abstractGraph(self.featsAtomListGraph,self.featsFamilyList)


        #(3) allocate memory spaces for nodeSPhGs_temp and edgeSPhGs_temp
        nodeSPhGs_temp = np.zeros((j*4,nCF*4),dtype=np.int32)#num of junctions is nCF-2
        edgeSPhGs_temp = np.zeros((j*4,nCF*4),dtype=np.int32)#num of junctions is nCF-2

        #(4) calculate (S)PhGs (nodeSPhGs and edgePHcts)
        iphct=0
        for i,combiFeats in enumerate(combinations(list(range(nfeats)),nCF)):
            if phg: nodeSPhGs_temp,edgeSPhGs_temp,iphct = self.__getNodeEdgePhG(nodeSPhGs_temp,edgeSPhGs_temp,combiFeats,nCF,iphct,verbose=verbose)
            else: nodeSPhGs_temp,edgeSPhGs_temp,iphct = self.getNodeEdgeSPhG(nodeSPhGs_temp,edgeSPhGs_temp,combiFeats,nCF,iphct,verbose=False)

        #(5) restore nodeSPhGs and edgeSPhGs with a minimum memory space
        self.nodeSPhGs = nodeSPhGs_temp[0:iphct]
        self.edgeSPhGs = edgeSPhGs_temp[0:iphct,:]
        print(' total num of phcts wo. multiples: %d'%iphct)


    def nCr(self,n,r):
        f = factorial
        if n<r: return 0
        else: return int( f(n) / f(r) / f(n-r) )

    def __getNodeEdgePhG(self,nodeSPhGs_temp,edgeSPhGs_temp,combiFeats,nCF,iphct,verbose=False):
        featsFamilyList = [v for i,v in enumerate(self.featsFamilyList) if i in combiFeats]
        featsAtomList2=[]
        for idx in combiFeats: featsAtomList2.append(self.featsAtomListGraph[idx])
        distmatPhcg = np.zeros((nCF,nCF),int)
        for iCF in range(nCF):
            for jCF in range(nCF):
                if iCF > jCF:
                    distmatPhcg[iCF,jCF] = np.amin(np.amin(self.distmatGraph[np.asarray(featsAtomList2[iCF]),:][:,np.asarray(featsAtomList2[jCF])]))
                    distmatPhcg[jCF,iCF] = distmatPhcg[iCF,jCF]
                    if distmatPhcg[iCF,jCF]==0:
                        distmatPhcg[iCF,jCF]=-1
                        distmatPhcg[jCF,iCF]=-1
        nodeSPhG0,edgeSPhG0 = self.encodeSPhG(distmatPhcg,featsFamilyList,nCF,verbose=False)
        if unique_node_edge_SPhG(nodeSPhGs_temp,edgeSPhGs_temp,nodeSPhG0,edgeSPhG0):
            nodeSPhGs_temp[iphct,:] = nodeSPhG0
            edgeSPhGs_temp[iphct,:] = edgeSPhG0
            iphct+=1
            if verbose:
                print("nodeSPhG:",end='')
                print(nodeSPhG0)
                print("edgeSPhG:",end='')
                print(edgeSPhG0)
        return nodeSPhGs_temp,edgeSPhGs_temp,iphct

    def getNodeEdgeSPhG(self,nodeSPhGs_temp,edgeSPhGs_temp,combiFeats,nCF,iphct,verbose=False):
        #(1) calculate SPhGList
        SPhGList =  self.g.abstractSubGraph(combiFeats)

        #(2) loop for each subSPhG
        for sphg0 in SPhGList:

            #(2-1) uncapsule variables (distmat and subFeatsListForAtom) in sphg0
            distmat = sphg0.distmat
            subFeatsListForAtom = sphg0.featsListForAtom

            #(2-2) count # of edges and nodes
            nedges0=np.sum(np.sum(distmat!=0))/2
            nnodes0=len(subFeatsListForAtom)

            if verbose:
                print('distmat')
                print(distmat)
                print('subFeatsListForAtom')
                print(subFeatsListForAtom)

            #(2-3) Eliminate mols with esceptionally many edges or nodes
            if nedges0>nCF*4 and nnodes0>nCF*4:
                if verbose: print('skip this sphg due to too many edges (= very rare case)')
            elif np.sum(np.sum(distmat))==0:
                if verbose: print('skip this sphg due to no edges defined')
            else:
                #(2-4) convert (subFeatsListForAtom and distmat) to (nodeSPhG0 and edgeSPhG0)
                nodeSPhG0,edgeSPhG0 = self.encodeSPhG(distmat,subFeatsListForAtom,nCF,verbose=verbose)
                nnodes=np.sum(nodeSPhG0>0)
                nedges=len(edgeSPhG0)

                #(2-5) Detect an isolated atom -> Error
                adjmat = np.zeros((nnodes,nnodes),int)
                for iedge in range(nedges):
                        edgeHash = edgeSPhG0[iedge]
                        idx0 = int(edgeHash//10000)
                        edgeHash1 = edgeHash%10000
                        idx1 = int(edgeHash1//100)
                        dist = int(edgeHash1%100)
                        if dist==0 and idx0!=idx1: dist=-1
                        adjmat[idx0,idx1] = dist
                        adjmat[idx1,idx0] = dist
                for inodes in range(nnodes): 
                    if np.sum(np.abs(adjmat[:,inodes]))==0:
                        print(combiFeats)
                    assert np.sum(np.abs(adjmat[:,inodes]))>0
                
                #(2-6) add new node/edgeSPhG0 into node/edgeSPhGs_temp only if there are no multiplications
                if unique_node_edge_SPhG(nodeSPhGs_temp,edgeSPhGs_temp,nodeSPhG0,edgeSPhG0):
                    #try:
                    if iphct < nodeSPhGs_temp.shape[0]: # skip this sphg if too many sphgs generated from a combinations of CFs.(=very rare pattern)
                        nodeSPhGs_temp[iphct,:] = nodeSPhG0
                        edgeSPhGs_temp[iphct,:] = edgeSPhG0
                        iphct+=1
                    #except:
                    #    pass
                else:
                    if verbose: print('skip this phct due to the phct equal to a previous one.')
        return nodeSPhGs_temp,edgeSPhGs_temp,iphct

    def __getDistmatPhcg(self,nCF,distmatGraph,featsAtomList2):
        distmatPhcg = np.zeros((nCF,nCF),int)
        for iCF in range(nCF):
            for jCF in range(nCF):
                if iCF > jCF:
                    distmatPhcg[iCF,jCF] = np.amin(np.amin(distmatGraph[np.asarray(featsAtomList2[iCF]),:][:,np.asarray(featsAtomList2[jCF])]))
                    distmatPhcg[jCF,iCF] = distmatPhcg[iCF,jCF]
                    if distmatPhcg[iCF,jCF]==0:
                        distmatPhcg[iCF,jCF]=-1
                        distmatPhcg[jCF,iCF]=-1
        return distmatPhcg


    ##################################################################
    def getStats(self,nCF=6,phg=False,verbose=False,stat=0):
        """
        Return aveMolCompIndex, and aveMolSparseIndex for each xmol
        """
        # (1) get Chemical Features
        self.getChemicalFeats(phg=phg,verbose=False)
        nfeats = len(self.featsFamilyList)

        # (2) skip small mols with fewer CFs than nCF
        if nfeats < nCF:
            print("skip the mol due to nfeats < nCF")
            return 0.0, 0.0
        else:
            
            j = self.nCr(nfeats,nCF)
            print(' molid: %s'%self.molid)
            print(' num of combinations: %d'%j)
            molCompIndexList=np.zeros(j,float)
            molSparseIndexList=np.zeros(j,float)
        
            # calc a whhole mol-sphg and save it in a self.g instance
            self.g.abstractGraph(self.featsAtomListGraph,self.featsFamilyList)
            for icombi,combiFeats in enumerate(combinations(list(range(nfeats)),nCF)):
        
                # calc PhG for confirming the distance preservation of SPhG
                featsFamilyList = [v for i,v in enumerate(self.featsFamilyList) if i in combiFeats]
                featsAtomList2=[]
                for idx in combiFeats: featsAtomList2.append(self.featsAtomListGraph[idx])
                distmatPhcg = np.zeros((nCF,nCF),int)
                for iCF in range(nCF):
                    for jCF in range(nCF):
                        if iCF > jCF:
                            distmatPhcg[iCF,jCF] = np.amin(np.amin(self.distmatGraph[np.asarray(featsAtomList2[iCF]),:][:,np.asarray(featsAtomList2[jCF])]))
                            distmatPhcg[jCF,iCF] = distmatPhcg[iCF,jCF]
                            if distmatPhcg[iCF,jCF]==0:
                                distmatPhcg[iCF,jCF]=-1
                                distmatPhcg[jCF,iCF]=-1
                nodeSPhG0,edgeSPhG0 = self.encodeSPhG(distmatPhcg,featsFamilyList,nCF,verbose=False)

                # calc SPhG
                SPhGList =  self.g.abstractSubGraph(combiFeats,convertAromaticToBond=False)
        
                compIndexList=[]
                sparceIndexList=[]
                for sphg0 in SPhGList:
        
                    # (1) calc sparceIndex
                    nodeSPhG1,edgeSPhG1 = self.encodeSPhG(sphg0.distmat,sphg0.featsListForAtom,nCF,verbose=False)
                    sparceIndexList.append( (np.sum(edgeSPhG1>0)+1)  /np.sum(nodeSPhG1>0))
        
                    # (2) calc compIndex(=bool)
                    # (2-1) calc all shortest distances in sphg0
                    ndimDistmat=sphg0.distmat.shape[0]
                    distmatCompleteGraph=np.zeros((ndimDistmat,ndimDistmat),np.int8)
                    g = Graph(sphg0.distmat.shape[0])
                    g.graphnp=sphg0.distmat
                    for iCF in range(ndimDistmat):
                        distvec, paths = g.shortestPaths(iCF)
                        distmatCompleteGraph[iCF,:] = distvec
        
                    # (2-2) calc PhG from sphg0
                    atomListForFeats_woJunction = []
                    featsListMol_woJunction=[]
                    for feats, atomList in zip(sphg0.featsListMol,sphg0.atomListForFeats):
                        if feats != 9:
                            atomListForFeats_woJunction.append(atomList)
                            featsListMol_woJunction.append(feats)
        
                    assert len(atomListForFeats_woJunction) == nCF
                    assert len(featsListMol_woJunction) == nCF
                    distmatPhcg0 = self.__getDistmatPhcg(nCF,distmatCompleteGraph,atomListForFeats_woJunction)

                    # (2-3) encode (featsListMol_woJunction, distmatPhcg0) to (nodeSPhG2, edgeSPhG2)
                    nodeSPhG2,edgeSPhG2 = self.encodeSPhG(distmatPhcg0,featsListMol_woJunction,nCF,verbose=False)
        
                    # (2-4) compare the PhG from the original mol and the PhG from sphg0
                    d1=nodeSPhG0 - nodeSPhG2
                    d2=edgeSPhG0 - edgeSPhG2
                    if np.sum(np.abs(d1)) == 0 and np.sum(np.abs(d2)) == 0: #assert np.sum(np.abs(d2))==0
                        compIndexList.append(True)
                    else:
                        compIndexList.append(False)
        
                # compIndex=True if there is at least one sphg0 reproducing node/edgeSPhG0 calculated from the original mol
                compIndex=False
                for bool0 in compIndexList:
                    if bool0: compIndex=True
        
                # sparseIndex are averaged with the whole sphg0
                sparceIndex = np.mean(np.asarray(sparceIndexList))
        
                molCompIndexList[icombi]=compIndex
                molSparseIndexList[icombi]=sparceIndex
                  
        
            aveMolCompIndex   = np.mean(np.asarray(molCompIndexList))
            aveMolSparseIndex = np.mean(np.asarray(molSparseIndexList))
            print('aveMolCompIndex: %7.5f,aveMolSparseIndex: %7.5f'%(aveMolCompIndex,aveMolSparseIndex),end='')
            return aveMolCompIndex, aveMolSparseIndex



    ###############################################
    def show(self,grid=True,num=True,highlight=[]):
        """
        Make an image for the xmol with chemical feature information
        """
        # calc distance matrix from an atom to another    
        AllChem.Compute2DCoords(self.mol)

        FeatFamilyList=list(self.FeatFamilyDict.keys())
        self.legends_list=[]

        #print(FeatFamilyList)
        #print(self.featsFamilyList)
        for featnum in self.featsFamilyList:
            self.legends_list.append(FeatFamilyList[featnum-1])

        featsAtomList_short = []
        legends_list_short = []
        for featFamily1 in FeatFamilyList:
            atomlist=[]
            for i,featFamily2 in enumerate(self.legends_list):
                if featFamily1==featFamily2:
                    if not(featFamily1 in legends_list_short): legends_list_short.append(featFamily1)
                    atomlist.extend(self.featsAtomListGraph[i])
            if len(atomlist)>0:featsAtomList_short.append(atomlist)

        if grid == True:
            im = Draw.MolsToGridImage([self.mol]*len(featsAtomList_short), legends=legends_list_short,\
                                      highlightAtomLists=featsAtomList_short)
            return im

        elif grid == False and num == True and len(highlight)>0:
            im = Draw.MolsToGridImage([self.mol]*len(highlight), legends=["hoge"]*len(highlight),\
                                      highlightAtomLists=highlight)
            return im

        elif grid == False and num == True and len(highlight)==0:
            # The option to include atomic numbers does not work!
            #Draw.DrawingOptions.includeAtomNumbers=True
            Draw.DrawingOptions.includeAtomNumbers=True
            #im = Draw.MolToImage(self.mol,includeAtomNumbers=True)
            Draw.MolToMPL(self.mol) 
        elif grid == False and num == False:
            Draw.MolToMPL(self.mol) 

@njit
def unique_node_edge_SPhG(nodeSPhGs_temp,edgeSPhGs_temp,nodeSPhG0,edgeSPhG0):
    """
    numba function to accelerate reduction of multiplicated SPhGs.

    """
    #for iphct, nodePhtvec edgeSPhGmat in zip(nodeSPhGs_temp,edgeSPhGs_temp):
    flagUnique=True
    for iphct, nodeSPhG in enumerate(nodeSPhGs_temp):
        if np.sum(nodeSPhG) ==0: break

        flagUnique=False
        for node0, node1 in zip(nodeSPhG0,nodeSPhG):
            if node0 != node1: flagUnique=True
        if flagUnique==False:
            edgeSPhG=edgeSPhGs_temp[iphct,:]
            for edge0, edge1 in zip(edgeSPhG0,edgeSPhG):
                if edge0 != edge1: flagUnique=True

        if flagUnique==False: break
    return flagUnique
  
