#!/bin/bash
./clean_testdir.sh
pytest test_graph.py
pytest test_ged.py
pytest test_sphg.py
pytest test_canonicalize.py
pytest test_buildDatabase.py
pytest test_analyzeAllCompounds.py
