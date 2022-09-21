from sensitivity_analysis import Sensitiviy_Analysis
from parsers import KParseArgs
import numpy as np
import networkx as nx
import sys
from graphs import KGraph
from preapre_data import KData
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle


class Reconstruction():

    def __init__(self) -> None:
        pass

    def construct_graph(self,args):
        method = Sensitiviy_Analysis()
        prepare_data = KData()
        data = args.data
        pairs = prepare_data.prepare_data(data)
        df = pd.read_csv('data/'+data)
        nodes = list(df.columns)
        predicted_matrix = []

        scores = []

        for node in nodes:
            predicted_labels,results = method.apply_method(args,pairs,int(node))
            scores.append(results)
            predicted_labels = list(predicted_labels)
            predicted_labels.insert(int(node), 0)
            #predicted_labels = np.array(predicted_labels)
            predicted_matrix.append(predicted_labels)
        #np.save('predictions/predicted_scores',np.array(scores))
        
        return np.array(predicted_matrix),np.array(scores)

    def symmetrize(self,args,predicted_matrix,scores):
        new_scores = []
        for i in range(args.node_size):
            new_scores.append(np.insert(scores[i],i,1.0))
        for i in range(predicted_matrix.shape[0]):
            for j in range(predicted_matrix[i].shape[0]):
                if predicted_matrix[i][j]!=predicted_matrix[j][i]:
                    mean_score = (new_scores[i][j]+new_scores[j][i])/2
                    if mean_score>=(new_scores[i].mean()+new_scores[j].mean())/2:
                        predicted_matrix[i,j]=1
                        predicted_matrix[j,i]=1
                    else:
                        predicted_matrix[i,j]=0
                        predicted_matrix[j,i]=0
        return predicted_matrix

    def reconstruct(self,args):
        predicted_matrix,scores = self.construct_graph(args)
        symmetric_predicted_matrix = self.symmetrize(args,predicted_matrix,scores)
        #np.save('predictions/predicted_matrix',predicted_matrix)
        
        return symmetric_predicted_matrix

    def ground_truth_graph(self,args,pred_matrix):
        graph_generator = KGraph()
        graph =  graph_generator.generate_graph(args)
        pred_graph = nx.from_numpy_matrix(pred_matrix)
        pred_graph = nx.convert_node_labels_to_integers(pred_graph)
        

        nx.draw(pred_graph, with_labels=True, pos=nx.kamada_kawai_layout(graph))
        plot_path = 'plots/reconstructed_graphs'
        if not os.path.exists('plots'):
            os.mkdir('plots')
            os.mkdir('plots/reconstructed_graphs')
            plt.savefig(plot_path+'/'+str(args.data)[5:]+'_'+str(args.graph)+str(args.node_size)+'_reconstructed.png')
        elif not os.path.exists('plots/reconstructed_graphs'):
            os.mkdir('plots/reconstructed_graphs')

            plt.savefig(plot_path+'/'+str(args.data)[5:]+'_'+str(args.graph)+str(args.node_size)+'_reconstructed.png')
        else:
            plt.savefig(plot_path+'/'+str(args.data)[5:]+'_'+str(args.graph)+str(args.node_size)+'_reconstructed.png')
        
        
        return graph


if __name__=="__main__":
        
    
    parser = KParseArgs()
    args = parser.parse_args()
    flag = len(sys.argv) == 1
    reconstructor = Reconstruction()
    symmetric_predicted_matrix = reconstructor.reconstruct(args)
    predicted_graph = reconstructor.ground_truth_graph(args,symmetric_predicted_matrix)





        
