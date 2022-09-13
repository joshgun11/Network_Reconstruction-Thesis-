from cmath import sqrt
import networkx as nx
import numpy as np
import math

class KGraph():

    def __init__(self) -> None:
        pass

    def generate_graph(self,args):
        NUM_NODES = args.node_size
        if args.graph=='Grid':
            
            alpha = int(math.sqrt(NUM_NODES))
            
            G =  nx.grid_2d_graph(alpha,alpha)
            g = nx.convert_node_labels_to_integers(G)
            return g

        elif args.graph=='Geom':
            G_geom_small = nx.random_geometric_graph(25, 0.3, seed=43)
            geom = nx.convert_node_labels_to_integers(G_geom_small)
            return geom

        elif args.graph=='Albert':
            albert = nx.barabasi_albert_graph(NUM_NODES, 2, seed=0)
            nx.draw(albert, with_labels=True, pos=nx.spring_layout(albert, seed=4))
            return albert

        elif args.graph=='Erdos':
            G_erdos_small = nx.erdos_renyi_graph(25, 0.15, seed=43)
            erdos = nx.convert_node_labels_to_integers(G_erdos_small)
            return erdos

