"""
Calculate scaffold-hopping performances. The directory prepared with build-database process is required
"""
import pytest
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.app as app
import pickle
import shutil

class AllCompoundsAnalyzer():
    """
    Make a set of SPhGs by analyzing all compunds with its actitivies info. as a train-pos/neg.
    Note that test-pos/neg.csv are the copies of train-pos.neg.
    """

    def run(self,tid,datadir,nsphg,ver,threshold=6.0,seed=99):
        """
        Special usecase for mapping SPhGs
        - (STEP2) Mining with all positives from ChEMBL, not a part of scaffold-types
          train-pos = test-pos, train-neg = test-neg
        - (STEP3) Select top-300 SPhGs with Cov, Nscf, and GR.
        - (STEP4) Calc Hitvecs.
        - After this function processed: analyze SPhGs for each tid with GTM, nanoscale or other mapping method
        """
        print("\n#### st01 ")
        app.st01_makeDataset.run(tid,ver,datadir,ntraintest=0,threshold=threshold,seed=seed)

        print("\n#### st02 ")
        res_mineSPhG_list1 = app.st02_mineSPhG.run(tid,ver,datadir)
        MineNodeSPhGs, MineEdgeSPhGs, MineHitvecs, SearchHitvecs_train_neg = res_mineSPhG_list1
        with open(datadir+ver+'/'+'st02_res_mineSPhG.pickle', mode='wb') as f: pickle.dump(res_mineSPhG_list1, f)

        print("\n#### st03 ")
        with open(datadir+ver+'/'+'st02_res_mineSPhG.pickle', mode='rb') as f: res_mineSPhG_list2 = pickle.load(f)
        [MineNodeSPhGs, MineEdgeSPhGs, MineHitvecs, SearchHitvecs_train_neg] = res_mineSPhG_list2
        [SelectedNodeEdgeSPhGsCov,SelectedNodeEdgeSPhGsScf,SelectedNodeEdgeSPhGsGR] = \
             app.st03_selectSPhG.run(tid,ver,datadir,MineNodeSPhGs, MineEdgeSPhGs, MineHitvecs, SearchHitvecs_train_neg,nsphg=nsphg) 
        res_selectSPhG_list3 = [SelectedNodeEdgeSPhGsCov,SelectedNodeEdgeSPhGsScf,SelectedNodeEdgeSPhGsGR] 
        with open(datadir+ver+'/'+'st03_res_selectSPhG.pickle', mode='wb') as f: pickle.dump(res_selectSPhG_list3, f)
 
        print("\n#### st04 ")
        ## search with selecteSPhGs 
        with open(datadir+ver+'/'+'st03_res_selectSPhG.pickle', mode='rb') as f: res_selectSPhG_list3 = pickle.load(f)
        [SelectedNodeEdgeSPhGsCov,SelectedNodeEdgeSPhGsScf,SelectedNodeEdgeSPhGsGR] = res_selectSPhG_list3
 
        # return object is a list of three select methods with Cov, Scf, and GR,
        # each of which is a dictionary of 4 two-dim-numpy-array SearchHitvecs
        SearchHitvecs_selectMethods,SearchHitsmiles_selectMethods = app.st04_searchSPhG.run(tid,ver,datadir,res_selectSPhG_list3) 
        res_searchSPhG_list4 = [SearchHitvecs_selectMethods,SearchHitsmiles_selectMethods]
        with open(datadir+ver+'/'+'st04_%d_res_searchSPhG.pickle'%nsphg, mode='wb') as f:
            pickle.dump(res_searchSPhG_list4, f)
