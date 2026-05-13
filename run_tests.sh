#! /bin/bash

time python3 run.py atomic 256 256 1 O 1
time python3 run.py atomic 256 256 1 O 4
time python3 run.py atomic 256 256 4 O 1
time python3 run.py atomic 256 256 4 O 4

time python3 run.py atomic 256 512 1 O 1
time python3 run.py atomic 256 512 1 O 4
time python3 run.py atomic 256 512 4 O 1
time python3 run.py atomic 256 512 4 O 4
