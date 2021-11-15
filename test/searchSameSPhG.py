import pytest
import os,sys
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.infra.search as search


#(1) retrieve SPhGs
def retrieveSPhGs(tid,seed,scaffoldtype,ntraintest):
    datadir = datadirroot + 'tid-%d/'%tid
    ver = 't%ds%d%sntt1%d'%(tid,seed,scaffoldtype,ntraintest)
    print(datadir+ver)
    with open(datadir+ver+'/'+'st03_res_selectSPhG.pickle', mode='rb') as f: res_selectSPhG_list3 = pickle.load(f)
    [_1,SelectedNodeEdgeSPhGsNScf,_2] = res_selectSPhG_list3
    return SelectedNodeEdgeSPhGsNScf

    

def searchSameSPhG(datadirroot,tid0,seed0,scaffoldtype0,ntraintest0,tid1,seed1,scaffoldtype1,ntraintest1):

    #(1) retrieve SPhGs
    SelectedNodeSPhGs0, SelectedEdgeSPhGs0 = retrieveSPhGs(tid0,seed0,scaffoldtype0,ntraintest0)
    SelectedNodeSPhGs1, SelectedEdgeSPhGs1 = retrieveSPhGs(tid1,seed1,scaffoldtype1,ntraintest1)

    assert len(SelectedNodeSPhGs0) == 300
    assert len(SelectedEdgeSPhGs0) == 300
    assert len(SelectedNodeSPhGs1) == 300
    assert len(SelectedEdgeSPhGs1) == 300

    #(2) calc hitmat
    search0 = search(SelectedNodeSPhGs0, SelectedEdgeSPhGs0)
    SearchHitvecs = search0.search([SelectedNodeSPhGs1], [SelectedEdgeSPhGs1])
    return SearchHitvecs


if __name__ == '__main__': 
    #tid0=int(sys.argv[1])
    #tid1=int(sys.argv[2])
    datadirroot = '../../data/sphg/'
    ntraintest0 = 0#int(sys.argv[3])
    ntraintest1 = 0#int(sys.argv[3])
    scaffoldtype0='bm'#sys.argv[6]#'ccrsingle' or 'ccrrecap' or 'bm'
    scaffoldtype1='bm'#sys.argv[6]#'ccrsingle' or 'ccrrecap' or 'bm'
    seed0=99
    seed1=99

    for tid0 in [8, 11, 137, 11362,20174,104219]:
        for tid1 in [8, 11, 137, 11362,20174,104219]:
            if tid0 > tid1:
                print('# Search shared SPhG in tid%d and tid%d'%(tid0,tid1))
                SearchHitvecs=searchSameSPhG(datadirroot,tid0,seed0,scaffoldtype0,ntraintest0,tid1,seed1,scaffoldtype1,ntraintest1)
                #print(SearchHitvecs.shape)
                #print(SearchHitvecs)
                print('Num of the shared SPhGs is')
                print(np.sum(np.array(SearchHitvecs)))
    

