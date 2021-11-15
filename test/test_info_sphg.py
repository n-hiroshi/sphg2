import pytest
import os,sys
import pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from sphg.common.sphgviewer import sphgviewer

def showSPhGs(datadir,ver,tid,saveImage=False,phg=False):

    # SPhGs Visualization
    with open(datadir+ver+'/'+'st03_res_selectSPhG.pickle', mode='rb') as f: res_selectSPhG_list3 = pickle.load(f)
    SelectedNodeEdgeSPhGs = res_selectSPhG_list3[1]# NScf
    SelectedNodeSPhGs = SelectedNodeEdgeSPhGs[0]
    SelectedEdgeSPhGs = SelectedNodeEdgeSPhGs[1]
    nsphg=len(SelectedNodeSPhGs)
    assert len(SelectedEdgeSPhGs) == nsphg

    if phg: head = 'phg'
    else: head = 'sphg'
    sv0=sphgviewer()   
    ## row layout
    for isphg in range(nsphg):
        print('# (S)PhG%d'%isphg)
        print(SelectedNodeSPhGs[isphg,:])
        print(SelectedEdgeSPhGs[isphg,:])
        
        if saveImage:
            sv0.show(SelectedNodeSPhGs[isphg,:],SelectedEdgeSPhGs[isphg,:],title='sphg%d'%isphg,figsize=(6,4),savename='%s%d_%03d.png'%(head,tid,isphg))
    

#if __name__ == '__main__': 
def test_map():
    tid=104219#int(sys.argv[1])
    phg=False
    if phg: datadir = './data/phg/tid-%d/'%tid
    else: datadir = './data/sphg/tid-%d/'%tid
    threshold =6.0#float(sys.argv[2])
    if tid==104219: threshold = 5.0
    ntraintest = 0#int(sys.argv[3])
    scaffoldtype='bm'#sys.argv[6]#'ccrsingle' or 'ccrrecap' or 'bm'
    seed=99
    ver = 't%ds%d%sntt1%d'%(tid,seed,scaffoldtype,ntraintest)
    showSPhGs(datadir,ver,tid,saveImage=True,phg=phg)
    

