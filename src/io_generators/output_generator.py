import matplotlib.pyplot as plt
from city_utils import *
import networkx as nx
import random
import sys
from random_utils import *

def gen_in(cities, numCities, homes, numHomes, start):
    filenamein = str(numCities) + ".in"
    with open(filenamein, "w+") as file:
        file.write(numCities)
        file.write(numHomes)
        file.write(cities)
        file.write(homes)
        file.write(start)

        #Graph



def gen_out(numCities, numHomes, lenCycle, numDropoffs):
    filenameout = str(numCities) + ".out"
    
    allCities = choose_cities(numCities)
    cycle = random.sample(allCities.split(), k=lenCycle)
    homes = random.sample(allCities.split(), k=numHomes)
    
    print('All cities:', allCities)
    print('Cycle:', cycle)
    print('Homes:', homes)
    homesCopy = list(homes)
    cycleCopy = list(cycle)
    random.shuffle(cycleCopy)
    
    dropoffStrings = []
    element_holder = [[] for _ in range(numDropoffs)]

    # distribute total_elements elements in randomly-sized portions among num_of_holders 'element holders' 
    random_portions_in_holder = RandIntVec(numDropoffs, numHomes, Distribution=RandFloats(numDropoffs))

    # assign each portion of elements to each 'holder'
    for h, portion in enumerate(random_portions_in_holder):
        for p in range(portion):
            index = random.randrange(len(homesCopy))
            element_holder[h].append(homesCopy.pop(index))

    for i in range(numDropoffs):
        dropoff = cycleCopy[i]
        taCities = element_holder[i]
        if dropoff in homes and dropoff not in taCities:
                restructure_holder(element_holder, dropoff, i)
        out = dropoff + ' '.join(taCities)
        dropoffStrings.append(out)

    with open(filenameout, "w+") as file:
        file.write(' '.join(cycle + [cycle[0]])  + " \n")
        file.write(str(numDropoffs) + " \n")
        for dropoff in dropoffStrings:
            file.write(dropoff  + " \n")


def restructure_holder(holders, dropoff, idx):
    for i, holder in enumerate(holders):
        if dropoff in holder and i != idx:
            holder.remove(dropoff)
            holders[idx].append(dropoff)
            break

if __name__ == '__main__':
    for i in [40, 100, 169]:
        # print(i)
        numHomes = int(input("Number of homes: "))
        lenCycle = int(input("Length of cycle: "))
        numDropoffs = int(input("How many dropoffs: "))
        gen_out(i, numHomes, lenCycle, numDropoffs)
