import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import numpy as np
from simanneal import Annealer # pip install simanneal

import random
from collections import defaultdict
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    G, message = adjacency_matrix_to_graph(adjacency_matrix)
    startLocation = list_of_locations.index(starting_car_location)
    homes = convert_locations_to_indices(list_of_homes, list_of_locations)

    # SHORTEST_PATHS = defaultdict(list)
    SHORTEST_PATHS_LENGTHS = defaultdict(list)
    for node in G.nodes:
        # [SHORTEST_PATHS[node].append(nx.shortest_path(G, source=node, target=n, weight='weight')) for n in G.nodes]
        [SHORTEST_PATHS_LENGTHS[node].append(nx.algorithms.shortest_path_length(G, source=node, target=n)) for n in G.nodes]

    sp_car, sp_dropoff = shortest_paths_solver(G, list_of_locations, homes, startLocation)
    cl_car, cl_dropoff = cluster_solver(G, list_of_locations, homes, startLocation, SHORTEST_PATHS_LENGTHS)

    cost_sp, message_sp = cost_of_solution(G, sp_car, sp_dropoff)
    cost_cl, message_cl = cost_of_solution(G, cl_car, cl_dropoff)

    if cost_sp < cost_cl:
        return sp_car, sp_dropoff
    else:
        return cl_car, cl_dropoff
    # if 'shortest_paths' in params:
    #     return shortest_paths_solver(G, list_of_locations, homes, startLocation)
    # elif 'cluster' in params:
    #     return cluster_solver(G, list_of_locations, homes, startLocation, SHORTEST_PATHS, SHORTEST_PATHS_LENGTHS)
    # elif 'anneal' in params:
    #     return anneal_solver(G, list_of_locations, homes, startLocation, SHORTEST_PATHS, SHORTEST_PATHS_LENGTHS)

def cluster_solver(G, list_of_locations, home_indices, starting_index, shortest_path_lengths):
    bestCost = float('inf')
    bestPath = None
    bestDropoff = None
    for _ in range(2):
        for numClusters in range(1, len(home_indices) + 1):
            mutableHomes = set(home_indices)
            # print('num_clusters', numClusters)
            centroids = findCentroids(G, random.sample(mutableHomes, k=numClusters), 200, shortest_path_lengths)
            car_path, dropoff = shortest_paths_solver(G, list_of_locations, centroids, starting_index)
            for drop in dropoff:
                dropoff[drop] = []
            for drop, homes in dropoff.items():
                for node, centroid in list(G.nodes(data='centroid')):
                    if centroid == drop and node in home_indices:
                        homes.append(node)
            dropoff = {k: v for k, v in dropoff.items() if len(v) > 0}
            cost, message = cost_of_solution(G, car_path, dropoff)
            if cost < bestCost:
                bestCost = cost
                bestPath = car_path
                bestDropoff = dropoff
    return bestPath, bestDropoff

def findCentroids(G, inital_centroids, iter_lim, shortest_path_lengths):
    centroids = inital_centroids
    iter_num = 0
    while iter_num < iter_lim:
        # print(iter_num)
        # print(centroids)
        iter_num += 1
        distances = [[(cent,  shortest_path_lengths[n][cent]) for cent in centroids] for n in G.nodes]
        closest_centroid = [min(dist, key=lambda d: d[1])[0] for dist in distances]
        d = defaultdict(list)
        for i, x in enumerate(closest_centroid):
            d[x].append(i)
        newCentroids = []
        for group in d.keys():
            nodeLengths = {}
            for member in d[group]:
                pathLengths = [shortest_path_lengths[member][node] for node in d[group]]
                nodeLengths[member] = sum(pathLengths)
            newCentroid = min(nodeLengths, key=nodeLengths.get)
            newCentroids.append(newCentroid)
        if newCentroids == centroids or iter_num >= iter_lim:
            nodes = [n for n in G]  # the actual id of the nodes
            cent_dict = {nodes[i]: closest_centroid[i] for i in range(len(nodes))}
            nx.set_node_attributes(G, cent_dict, 'centroid')
            break
        else:
            centroids = newCentroids
    return centroids


def shortest_paths_solver(G, list_of_locations, home_indices, starting_index):
    mutableHomes = list(home_indices)
    currLocation = starting_index
    car_path = [currLocation]
    dropoffs = {}

    while len(mutableHomes) > 0:
        shortestPathLength = np.array([nx.algorithms.shortest_path_length(G, source=currLocation, target=home) for home in mutableHomes])
        argMin = np.argmin(shortestPathLength)
        next_loc = mutableHomes[argMin]
        shortestPath = nx.algorithms.shortest_path(G, source=currLocation, target=next_loc)
        car_path.extend(shortestPath[1:])
        dropoffs[next_loc] = [next_loc]
        currLocation = next_loc
        mutableHomes.pop(argMin)
    car_path.extend(nx.algorithms.shortest_path(G, source=currLocation, target=starting_index)[1:])
    return car_path, dropoffs

def which_dropoff(route, home):
    """
    Where should a TA get dropped off along Rao's route?
    route: driving cycle (list)
    home: the TA that's getting dropped off
    """
    return min(route, key=lambda i: SHORTEST_PATHS_LENGTHS[i][home])


def anneal_solver(G, list_of_locations, homes, startLocation, shortest_paths, shortest_path_lengths):
    # state is [Rao's route]
    route = [startLocation] + [i[1] for i in nx.find_cycle(G, source=startLocation)]
    # dropoffs = {}
    # for i in range(len(route) - 1):
    #     dropoffs[route[i]] = homes[i // len(route) * len(homes) : (i + 1) // len(route) * len(homes)]
    init_state = [route]
    dth = DTH(init_state, G, startLocation)
    itinerary, e = dth.anneal()
    dropoffs = defaultdict(list)
    for home in self.homes:
        d = which_dropoff(self.state, home)
        dropoffs[d].append(home)
    return itinerary, dropoffs


# Simulated Annealing
class DTH(Annealer):

    # Pass in initial state and graph
    def __init__(self, state, graph, start, homes):
        self.graph = graph
        self.start = start
        self.homes = homes
        super(DTH, self).__init__(state)  # important!

    def move(self):
        """Creates next candidate state, returns change in energy"""
        initial = self.energy()

        if len(self.state) == 1:
            # Add a city to route
            next = random.choice([b for b in self.graph[self.start]])
            self.state.extend([next, self.start])
        else:
            r = random.random()
            if r < 0.5:
                # add city
                pass
            else:
                # remove city
                pass


        # if r < 0.25:
        #     # if self.state[1].keys() == 1
        #     # Move one home from one dropoff to another
        #     drop1 = random.choice(self.state[0])
        #     while self.state[1][drop1] == []:
        #         # ensures there are dropoffs at this location
        #         drop1 = random.choice(self.state[0])
        #     drop2 = random.choice(self.state[0])
        #     while drop2 == drop1:
        #         drop2 = random.choice(self.state[0])
        #     tomove = random.choice(self.state[1][drop1])
        #     self.state[1][drop1].remove(tomove)
        #     self.state[1][drop2].append(tomove)
        # elif r < 0.5:
        #     # Add a city to Rao's route
        #
        #     self.state[1][toadd] = []
        # elif r < 0.75:
        #     # Remove a city from Rao's route, if it is a dropoff point, move them to another dropoff
        #     pass
        # else:
        #     #
        #     pass



        return initial - self.energy()

    def energy(self):
        """Calculates total cost of trip"""
        dropoffs = defaultdict(list)
        for home in self.homes:
            d = which_dropoff(self.state, home)
            dropoffs[d].append(home)
        return cost_of_solution(self.graph, self.state, dropoffs)

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
