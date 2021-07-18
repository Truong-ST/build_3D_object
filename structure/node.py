import numpy as np


class Node:
    def __init__(self, vertex, edge, color, parallel):
        self.vertex = vertex
        self.edge = edge
        self.color = color
        self.parallel = parallel
        self.link = []
        
        
    def connect(self, node):
        self.link.append(node)
        
        
    def infor(self):
        print(self.vertex, self.edge, self.color, self.parallel)
        
        
if __name__ == "__main__":
    n = Node(2,4,1,1)
    b = Node(1,3,0,0)
    n.connect(b)
    print(n.link[0].vertex)
    

