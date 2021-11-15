import pytest
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.app as app
import pickle
from sphg.common.sphgviewer import sphgviewer

#if __name__ == '__main__':
def test_analyzeAllCPDs():
    head='a00'
    ntraintest = 0 # 0 for analyze with All Compunds as a training data set
    nsphg=300
    seed = 99#int(sys.argv[3])
    tid = 104219# int(sys.argv[1])
    threshold =5.0#float(sys.argv[2])
    #datadir = './data/sphg/tid-%d/'%tid
    datadir = './data/phg/tid-%d/'%tid
    scaffoldtype='bm'
    ver = 't%ds%d%sntt1%d'%(tid,seed,scaffoldtype,ntraintest)
    print("\n##### %s #####"%ver)
    aca = app.AllCompoundsAnalyzer()
    aca.run(tid,datadir,nsphg,ver,threshold=threshold,seed=seed)
    
