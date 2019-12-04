import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import numpy as np

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

    if 'shortest_paths' in params:
        return shortest_paths_solver(G, list_of_locations, homes, startLocation)
    elif 'cluster' in params:
        return cluster_solver(G, list_of_locations, homes, startLocation)

def cluster_solver(G, list_of_locations, home_indices, starting_index):
    bestCost = float('inf')
    bestPath = None
    bestDropoff = None
    for numClusters in range(1, len(home_indices) + 1):
        mutableHomes = set(home_indices)
        centroids = findCentroids(G, random.sample(mutableHomes, k=numClusters), 300)
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

def findCentroids(G, inital_centroids, iter_lim):
    centroids = inital_centroids
    while iter_lim > 0:
        iter_lim -= 1
        shortest_paths = [[(cent, nx.shortest_path(G, source=n ,target=cent, weight='weight')) for cent in centroids] for n in G.nodes]
        distances = [[(sp[0],  sum([G[sp[1][i]][sp[1][i+1]]['weight'] for i in range(len(sp[1]) - 1)]) if len(sp[1]) > 1 else 0) for sp in sps] for sps in shortest_paths]
        closest_centroid = [min(dist, key=lambda d: d[1])[0] for dist in distances]
        d = defaultdict(list)
        for i, x in enumerate(closest_centroid):
            d[x].append(i)
        newCentroids = []
        for group in d.keys():
            nodeLengths = {}
            for member in d[group]:
                pathLengths = [nx.algorithms.shortest_path_length(G, source=member, target=node) for node in d[group]]
                nodeLengths[member] = sum(pathLengths)
            newCentroid = min(nodeLengths, key=nodeLengths.get)
            newCentroids.append(newCentroid)
        if set(newCentroids) == set(centroids) or iter_lim == 0:
            nodes = [n for n in G]  # the actual id of the nodes
            cent_dict = {nodes[i]: closest_centroid[i] for i in range(len(nodes))}
            nx.set_node_attributes(G, cent_dict, 'centroid')
            break
        else:
            centroids = newCentroids
    return centroids


def shortest_paths_solver(G, list_of_locations, home_indices, starting_index):
    currLocation = starting_index
    car_path = [currLocation]
    dropoffs = {}

    while len(home_indices) > 0:
        shortestPathLength = np.array([nx.algorithms.shortest_path_length(G, source=currLocation, target=home) for home in home_indices])
        argMin = np.argmin(shortestPathLength)
        next_loc = home_indices[argMin]
        shortestPath = nx.algorithms.shortest_path(G, source=currLocation, target=next_loc)
        car_path.extend(shortestPath[1:])
        dropoffs[next_loc] = [next_loc]
        currLocation = next_loc
        home_indices.pop(argMin)
    car_path.extend(nx.algorithms.shortest_path(G, source=currLocation, target=starting_index)[1:])
    return car_path, dropoffs


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
