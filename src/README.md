# project-fa19
CS 170 Fall 2019 Project

This project depends on libraries that are not part of the standard Python distribution, so run:

```
pip install numpy networkx simanneal
```
to install the proper libraries.

If you want to run solver on one input file, you will want to run:

```
python3 solver.py <path to input file> <path to output directory>
```

If you want to run solver on all of the input files in a directory, you will want to run:
```
python3 solver.py --all <path to input directory> <path to output directory>
```

Note: We changed the range in line 63 to be 2 for *_200.in, and 3 for *_100.in *_50.in.
The solver will run the different strategies and return the output with the lowest cost.
