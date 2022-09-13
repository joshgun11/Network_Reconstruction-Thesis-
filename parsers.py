import argparse

class KParseArgs():

    def __init__(self):
        self.args = parser = argparse.ArgumentParser()

        self.parser = argparse.ArgumentParser()
        

        

        self.parser.add_argument("--data", help="data_path", action='store', nargs='?', default='Voter_grid_10K.csv',
                            type=str)

        self.parser.add_argument("--epochs", help="epochs", action='store', nargs='?', default=200,
                            type=int)    

        self.parser.add_argument("--batch_size", help="batchs size", action='store', nargs='?', default=256,
                            type=int)    
        self.parser.add_argument("--node", help="traget node", action='store', nargs='?', default=1,
                            type=int)    
        self.parser.add_argument("--plot_cluster", help="plot or not clustering", action='store', nargs='?', default=False,
                            type=bool)   
        self.parser.add_argument("--graph", help="graph type", action='store', nargs='?', default='Grid',
                            type=str)  
        self.parser.add_argument("--node_size", help="graph type", action='store', nargs='?', default=25,
                            type=int)  
         
    
    
    def parse_args(self):
        return self.parser.parse_args()

    def parse_args_list(self, args_list):
        return self.parser.parse_args(args_list)