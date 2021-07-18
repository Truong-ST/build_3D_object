import numpy as np
import cv2 as cv


def compare_node(node1, node2):
    rs = node1 - node2
    return rs

def connect(node1, node2):
    pass

def connect_graph(graph1, graph2):
    const = 10
    for vertex in graph1:
        for v in graph2:
            if compare_node(vertex, v) > const:
                connect(vertex, v)
    pass


def build(graph):
    pass