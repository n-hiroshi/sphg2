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


if __name__ == '__main__':
#def test_buildDatabase():
    tid=int(sys.argv[1])
    datadir = os.path.dirname(os.path.abspath(__file__))
    datadir += '/data/%s/tid-%d/'%(sys.argv[2],tid)
    if sys.argv[2] == 'phg': phg=True
    elif sys.argv[2] == 'sphg': phg=False
    else: print('Illegal option in sys.argv[2]')
    os.makedirs(datadir+'xmoldbs/', exist_ok=True)
    bd=app.DatabaseBuilder()
    bd.parallel_getALlSPhGs(tid,phg,datadir)
    bd.convert_to_XmolDB(tid,datadir)

