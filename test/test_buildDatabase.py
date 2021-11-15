# -*- coding: utf-8 -*-
"""
Created on 2020/06/20

@author: Hiroshi Nakano
"""
import pytest
from rdkit import Chem
import os,sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import sphg.app as app


#if __name__ == '__main__':
def test_buildDatabase():

    tid=104219
    phg = True#False
    if phg:#PhG
        datadir = './data/phg/tid-%d/'%tid
    else: #SPhG
        datadir = './data/sphg/tid-%d/'%tid
    os.makedirs(datadir+'xmoldbs/', exist_ok=True)
    bd=app.DatabaseBuilder()
    bd.parallel_getALlSPhGs(tid,phg,datadir)
    bd.convert_to_XmolDB(tid,datadir)
    assert True

