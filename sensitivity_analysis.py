from multiprocessing.spawn import prepare
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from preapre_data import KData
from train import KTrain
from parsers import KParseArgs
import sys
import os

class Sensitiviy_Analysis():

    def __init__(self) -> None:
        pass

    def feature_importance(self,x,y,X_test,model):
        features = list(x.columns)
        predictions = model.predict(X_test)
        importances = {}
        for var in features:
            X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.33,random_state = 43,stratify = y)
            X_test[var] = np.random.permutation(X_test[var])
        
            var_predictions = model.predict(X_test)
        
            s_p = np.abs(predictions-var_predictions)
            s_p = s_p.sum()
            s_p = s_p/X_train.shape[0]
            importances[str(var)]=s_p
        return importances 

    def prepare_scores(self,scores_dict):
        results = []
        for i in scores_dict.values():
            results.append(i)
        results = np.array(results)
    
        return results

    def clustering(self,args,results,X,plot,node):
        cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
        cluster.fit_predict(results.reshape(-1, 1))
        #a = np.count_nonzero(cluster.labels_)
        labels = cluster.labels_
        cluster_data = pd.DataFrame({"label":labels,"score":results})
        neighbours_mean_score = cluster_data.loc[cluster_data["label"]==1]["score"].mean()
        non_neighbours_mean_score = cluster_data.loc[cluster_data["label"]==0]["score"].mean()
    
        if neighbours_mean_score<non_neighbours_mean_score:
            labels = np.where((labels==0)|(labels==1), labels^1, labels)
        if plot:
            plt.scatter(np.array(list(X.keys())),results, c=labels, cmap='rainbow')
            plot_path = 'plots/clustering'
            if not os.path.exists('plots'):
                os.mkdir('plots')
                os.mkdir('plots/clustering')
                plt.savefig(plot_path+'/'+str(args.data)[5:]+'_'+str(node)+'_node_clustering.png')
                
            else:
                
                plt.savefig(plot_path+'/'+str(args.data)[5:]+'_'+str(node)+'_node_clustering.png')
            

        return labels

    def apply_method(self,args,pairs,node):
        print('Process is going for node: '+str(node))
        train_model = KTrain()
        

        x,y,X_test,model = train_model.train(args,pairs,node)
        scores_dict = self.feature_importance(x,y,X_test,model)
        results = self.prepare_scores(scores_dict)
         
        labels = self.clustering(args,results,scores_dict,plot=args.plot_cluster,node = node)
        return labels,results

if __name__=="__main__":
    method = Sensitiviy_Analysis()
    parser = KParseArgs()
    prepare_Data = KData()
    args = parser.parse_args()

    flag = len(sys.argv) == 1
    pairs = prepare_Data.prepare_data(args.data)

    labels = method.apply_method(args,pairs,args.node)
    
    print(labels)









    

    