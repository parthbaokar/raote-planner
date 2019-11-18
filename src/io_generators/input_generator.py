import numpy as np
import networkx as nx

V = 100 # number of nodes
D = 2 # dimensionality

def generate(V, D):
    positions = np.random.rand(V, D)
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=-1)) # euclidean

    # create a weighted, directed graph in networkx
    graph = nx.from_numpy_matrix(distances, create_using=nx.DiGraph())
    filename = str(V) + ".out"
    file = open(filename, "w")
    for line in nx.generate_adjlist(graph):
        file.write(line)
    file.close()

if __name__ == '__main__':
    sizes = [50, 100, 200]
    D = 2
    for V in sizes:
        generate(V, D)
