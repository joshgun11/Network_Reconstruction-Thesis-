import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from train import KTrain
from parsers import KParseArgs
import sys
import os
class Sensitiviy_Analysis():

    def __init__(self) -> None:
        pass

    def feature_importance(self,X,y,model):

        results = {}
        

        for column in list(X.columns):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=7,stratify=y)
            X_test[column] = 0
            X_test = np.array(X_test)
            acc = model.evaluate(X_test,y_test)[1]
            preds = tf.keras.activations.sigmoid(model.predict(X_test))
            acc = acc*100
            acc = 100-acc
            results[column] = acc
        
        return results

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

    def apply_method(self,args,node):
        print('Process is going for node: '+str(node))
        train_model = KTrain()

        x,y,model = train_model.train(args,node)
        scores_dict = self.feature_importance(x,y,model)
        results = self.prepare_scores(scores_dict)
         
        labels = self.clustering(args,results,scores_dict,plot=args.plot_cluster,node = node)
        return labels

if __name__=="__main__":
    method = Sensitiviy_Analysis()
    parser = KParseArgs()
    args = parser.parse_args()

    flag = len(sys.argv) == 1

    labels = method.apply_method(args,args.node)
    
    print(labels)









    

    