# CS 170 Fall 2019 Project

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

The solver will run the different strategies and return the output with the lowest cost.

## Strategies
The two fully implemented strategies are:
1. Shortest paths: this strategy has Professor Rao optimally driving each TA to their home.
2. Clusters: this strategy finds an optimal number of centroids, then uses shortest paths to find the best way for Rao to get to all of the centroids.

The results from both of these strategies are compared, and the lower cost solution is returned.

A third strategy which we did not have time to finish is an implementation of simulated annealing, which is a probabilistic technique for approximating the global optimum of a given function. At each iteration of this algorithm, it considers accepting a new state with some probability, with this converging to a global optimum in the long run. We tried to use the API from simanneal with limited success.

Specifically, our approach was to assign Rao's driving route to our state. Each move, we either added or removed a location to his route. Sometimes, more than one location would be added because the neighbors of the insertion/deletion point were not connected. To calculate the energy, each TA chose the closest location on the route to be its dropoff point, and then the total cost was calculated.
