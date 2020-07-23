import numpy as np
import networkx


def print_graph(graph: networkx.Graph):
    weight_matrix = np.zeros((len(graph.nodes), len(graph.nodes)))
    for i, n1 in enumerate(graph.nodes):
        for j, n2 in enumerate(graph.nodes):
            try:
                dist = graph[n1][n2]["weight"]
            except KeyError:
                dist = 0
            weight_matrix[i, j] = dist
    print(weight_matrix)

