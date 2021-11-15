#!/bin/bash
rm data/sphg/tid-0/xmoldbs/*
rm data/sphg/tid-0/xmoldbs_tid0.pickle
rm -r data/sphg/tid-0/t0s0bmntt14/

rm -r data/sphg/tid-104219
mkdir data/sphg/tid-104219
cp data/tid-104219-master.csv data/sphg/tid-104219/

rm -r data/phg/tid-104219
mkdir data/phg/tid-104219
cp data/tid-104219-master.csv data/phg/tid-104219/
