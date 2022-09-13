import pandas as pd 
import numpy as np 


class KData():

    def __init__(self) -> None:
        pass

    
    def prepare_data(self,path):

        data = pd.read_csv('data/'+path)
        nodes = data.columns
        pairs = []
        for i in nodes:
            y = data[str(i)]
            y = y[1:]
            x = data.drop(str(i),axis=1)
            x = x[:-1]
            pairs.append((x,y))
    
        return pairs

    def single_experiment(self,path,node):

        pairs = self.prepare_data(path)
        x = pairs[node][0]
        y = pairs[node][1]

        return x,y


    