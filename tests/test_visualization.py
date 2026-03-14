import numpy as np

from src.utils.visualization import graph_to_dot


def test_graph_to_dot_output_contains_nodes_and_edges():
    # Create a simple graph observation with two nodes connected by an edge.
    obs = {"adj": np.array([[0.0, 1.0], [1.0, 0.0]]), "node": np.array([[1.0, 1.0], [-1.0, 1.0]])}
    dot = graph_to_dot(obs)

    assert "strict graph" in dot
    assert "n0 -- n1" in dot
    assert "n0" in dot
    assert "n1" in dot
