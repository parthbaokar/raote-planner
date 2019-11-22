import matplotlib.pyplot as plt
from city_utils import *
import networkx as nx
import random
import sys

def genin(cities, numcities, homes, numhomes, start):
    filenamein = str(numcities) + ".in"
    filein = open(filenamein, "w")
    filenamein.write(allcities)
    filenameout.write(' '.join(cycle))

def genout(numcities, numhomes, lencycle):
    filenameout = str(i) + ".out"
    fileout = open(filenameout, "w")

    allcities = choose_cities(numcities)
    cycle = random.sample(allcities.split(), k=lencycle)


if __name__ == '__main__':
    for i in [40, 100, 169]:
        
