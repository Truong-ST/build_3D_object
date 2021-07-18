import numpy as np


class CharacteristicGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        
    
    def connect_node(self, index1, index2, weight):
        node1 = self.nodes[index1]
        node2 = self.nodes[index2]
        node1.connect(node2)
        node2.connect(node1)
        self.edges.append([(index1, index2), weight])