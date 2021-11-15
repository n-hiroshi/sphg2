"""
Canonicalization of SPhGs with the method of [J. Chem. Inf. Model. 2015, 55, 2111–2120].
"""

import os,sys
import numpy as np
from numba import njit

class Canonicalizer:
    def __init__(self):
        pass
    def canonicalize(self,nodeSetList,edges):
        maxdist = 30#16

        nodes=[]
        for nodeSet in nodeSetList:
            if isinstance(nodeSet,int):
                node=nodeSet
            else:
                nodeSet.sort()
                node=0
                for v in nodeSet: node=node*10+v
            nodes.append(node)
        nodes = np.asarray(nodes)
        edges = np.asarray(edges)
        edges2 = np.vstack((edges,edges[:,[1,0,2]]))
        nnodes = len(nodes)
        ranks = np.ones(nnodes,int)
        done =  np.zeros(nnodes,int)

        ## (1) rank with CF-types
        priorityVec = nodes.copy()
        uniquePriorityVec = np.sort(np.unique(priorityVec))

        knodes=0
        for v in uniquePriorityVec:
            hitvec = (priorityVec==v)
            ranks[hitvec] = knodes
            knodes+=np.sum(hitvec)

        ## (2) rank with num of bonds and their distances 
        # calc priorityVec for all nodes
        pritorityVec = np.zeros((nnodes),int)
        for inode in np.arange(nnodes):
            bondvec = edges2[edges2[:,0]==inode,2]
            bondvec = np.sort(bondvec)
            priority=0
            for dist in bondvec: priority=priority*maxdist+dist
            priorityVec[inode] = priority

        # re-ranking in a each equivelent group with the priorityVec
        ranks = self.__getFinerRanks(ranks,priorityVec)


        ## (3-1) rank with the neibourhood atom ranks        
        # calc priorityVec for all nodes
        for iite in np.arange(nnodes):
            pritorityVec = np.zeros((nnodes),int)
            for inode in np.arange(nnodes):
                adjatomvec = edges2[edges2[:,0]==inode,1]
                adjatomRankVec = adjatomvec.copy()
                for iatom,adjatom in enumerate(adjatomvec):
                    adjatomRankVec[iatom] = ranks[adjatom]
                adjatomRankVec = np.sort(adjatomRankVec)
                priority=0
                for adjatomRanks in adjatomRankVec: priority=priority*nnodes+adjatomRanks
                priorityVec[inode] = priority

            # re-ranking in a each equivelent group with the priorityVec
            ranksNew = self.__getFinerRanks(ranks,priorityVec)
           
            if list(ranksNew) == list(ranks) and len(ranks) > len(np.unique(ranks)):
                ## (3-2) tie-break       
                ranksOld = ranks.copy()
                uniqueRanksOld = np.sort(np.unique(ranks))
                for u in uniqueRanksOld:
                    hitvec =  (ranksOld==u)
                    if np.sum(hitvec) > 1:
                        ranks[hitvec] = np.arange(np.sum(hitvec))+ranksOld[hitvec][0]
                        break
            elif list(ranksNew) == list(ranks) and len(ranks) == len(np.unique(ranks)):
                break
            else:
                ranks = ranksNew


        ## final update of nodes and edges
        nodesNew = nodes.copy()
        edgesNew = edges.copy()
        for inode in np.arange(nnodes):
           nodesNew[ranks[inode]] = nodes[inode]
        for iedge in np.arange(edges.shape[0]):
           nodeA = ranks[edges[iedge,0]]
           nodeB = ranks[edges[iedge,1]]
           if nodeA<nodeB:
              edgesNew[iedge,0] = nodeA
              edgesNew[iedge,1] = nodeB
           elif nodeA>nodeB:
              edgesNew[iedge,0] = nodeB
              edgesNew[iedge,1] = nodeA
           else: 
              print(nodeA)
              print(nodeB)
              assert False
        
        sortidx = edgesNew[:,0]*nnodes+edgesNew[:,1]
        argsortvec = np.argsort(sortidx)
        edgesNew = edgesNew[argsortvec,:]
        
        return nodesNew,edgesNew

    ### inner function for canonicalize()
    def __getFinerRanks(self,ranks,priorityVec):
        ranksOld = ranks.copy()
        uniqueRanksOld = np.unique(ranks)
        for u in uniqueRanksOld:
            priorityVec_sub=priorityVec[ranksOld==u]
            if len(priorityVec_sub) ==1:
                pass
            else:
                unique_priorityVec_sub = np.sort(np.unique(priorityVec_sub))
                knodes=u
                ranks_sub = ranksOld[ranksOld==u]
                for v in unique_priorityVec_sub:
                    hitvec = (priorityVec_sub==v)
                    #ranks_sub[priorityVec_sub==v]=knodes
                    ranks_sub[hitvec]=knodes
                    knodes+=np.sum(hitvec)
                ranks[ranksOld==u]=ranks_sub
        return ranks
