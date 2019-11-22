import matplotlib.pyplot as plt
from city_utils import *
import networkx as nx
import random
import sys

def genin(cities, numcities, homes, numhomes, start):
    filenamein = str(numcities) + ".in"
    with open(filenamein, "w+") as file:
        file.write(numcities)
        file.write(numhomes)
        file.write(cities)
        file.write(homes)
        file.write(start)

        #Graph



def genout(numcities, numhomes, lencycle, numdropoffs):
    filenameout = str(numcities) + ".out"
    allcities = choose_cities(nties.split(), k=numcities)
    cycle = random.sample(allcities.split(), k=lencycle)
    homes = random.sample(allcities.split(), k=numhomes)
    cyclecopy = list(cycle)
    random.shuffle(cyclecopy)
    dropoffstrings = []
    
    for i in range(numdropoffs):
        dropoff = cyclecopy[i]
        out = dropoff
        if dropoff in homes:
            out += " " + dropoff



    with open(filenameout, "w+") as file:
        file.write(' '.join(cycle + [cycle[0]]))




if __name__ == '__main__':
    for i in [40, 100, 169]:
        print(i)
        numhomes = int(input("Number of homes: "))
        lencycle = int(input("Length of cycle: "))
        numdropoffs = int(input("How many dropoffs: "))
        genout(i, numhomes, lencycle, numdropoffs)
