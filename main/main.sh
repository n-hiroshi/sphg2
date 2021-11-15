#!/bin/bash
#PBS -q longq
#PBS -N sphg
#PBS -l select=1:ncpus=20
cd $PBS_O_WORKDIR

python=$HOME/miniconda3/bin/python
main=$HOME/02_pharmacophore/sphg2/main
tid=104219

$python ${main}/main_buildDatabase.py $tid sphg > sphg1.log
$python ${main}/main_analyzeAllCompounds.py $tid sphg #> sphg2.log
$python ${main}/main_info_sphg.py $tid sphg > sphg.dat

$python ${main}/main_buildDatabase.py $tid phg > phg1.log
$python ${main}/main_analyzeAllCompounds.py $tid phg > phg2.log
$python ${main}/main_info_sphg.py $tid phg > phg.dat
