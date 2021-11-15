#!/bin/bash
#PBS -q longq
#PBS -N ged
#PBS -l select=1:ncpus=20
cd $PBS_O_WORKDIR
~/miniconda3/bin/python ~/02_pharmacophore/sphg2/test/main_makeGEDmat.py > map104219.log
