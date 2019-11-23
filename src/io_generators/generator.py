import matplotlib.pyplot as plt
from city_utils import *
import networkx as nx
import random
import sys
from random_utils import *

INPUT_DIR = '../../inputs/'
OUTPUT_DIR = '../../outputs/'

def gen_in(cycle, cities, numCities, homes, numHomes, start, dropoffStrings):
    # Graph
    G = nx.Graph()
    # cyclelist = cycle.split()
    first = cycle[0]
    done = set(cycle)
    # cycle
    for i in range(1, len(cycle)):
        G.add_edge(first, cycle[i], weight=5)
        first = cycle[i]

    # dropoffs - each walker has a direct edge to the dropoff location
    for row in dropoffStrings:
        dropofflist = row.split()
        done |= set(dropofflist)
        dropoff = dropofflist[0]
        for i in range(1, len(dropofflist)):
            if dropoff != dropofflist[i]:
                G.add_edge(dropoff, dropofflist[i], weight=6)

    # time to add the rest of the cities to our graph (randomly)
    # print(cities)
    # print(done)
    citylist = cities.split()
    for city in citylist:
        if city not in done:
            # Add to graph
            u = random.sample(done, k=1)[0]
            G.add_edge(u, city, weight=random.randint(5, 9))
            done |= {city}

    #add random edges
    for i in range(numCities):
        u = random.sample(citylist, k=1)[0]
        v = random.sample(citylist, k=1)[0]
        if u == v:
            v = random.sample(citylist, k=1)[0]
        G.add_edge(u, v,  weight = 8)

    # pos = nx.spring_layout(G)
    # nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # nx.draw_networkx_edge_labels(G, pos, font_size=20, font_family="sans-serif", edge_labels = nx.get_edge_attributes(G, 'weight'))



    filenamein = INPUT_DIR + str(numCities) + ".in"
    with open(filenamein, "w+") as file:
        file.write(str(numCities) + '\n')
        file.write(str(numHomes) + '\n')
        file.write(cities + '\n')
        file.write(' '.join(homes) + '\n')
        file.write(start + '\n')
        # l = []
        # for n1, n2, attr in G.edges(data=True):
        #     l.append(n1, n2, attr['weight'])
        for city1 in citylist:
            out = ""
            for city2 in citylist:
                try:
                    weight = G[city1][city2]['weight']
                except:
                    weight = "x"
                out += str(weight) + " "
            file.write(out + "\n")



        #Graph

def gen_out(numCities, numHomes, lenCycle, numDropoffs):
    filenameout = OUTPUT_DIR + str(numCities) + ".out"

    allCities = choose_cities(numCities)
    cycle = random.sample(allCities.split(), k=lenCycle)
    homes = random.sample(allCities.split(), k=numHomes)

    # print('All cities:', allCities)
    # print('Cycle:', cycle)
    # print('Homes:', homes)
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
        out = dropoff + ' ' + ' '.join(taCities)
        dropoffStrings.append(out)

    gen_in(cycle, allCities, numCities, homes, numHomes, cycle[0], dropoffStrings)

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
    for i in [50, 100, 200]:
        print("For {0} cities choose the following options".format(i))
        numHomes = int(input("Number of homes: "))
        lenCycle = int(input("Length of cycle: "))
        numDropoffs = int(input("How many dropoffs: "))
        gen_out(i, numHomes, lenCycle, numDropoffs)
