import argparse

class KParseArgs():

    def __init__(self):
        self.args = parser = argparse.ArgumentParser()

        self.parser = argparse.ArgumentParser()
        

        

        self.parser.add_argument("--data", help="data_path", action='store', nargs='?', default='grid_16_voter_5000.csv',
                            type=str)

        self.parser.add_argument("--epochs", help="epochs", action='store', nargs='?', default=100,
                            type=int)    

        self.parser.add_argument("--batch_size", help="batchs size", action='store', nargs='?', default=64,
                            type=int)    
        self.parser.add_argument("--node", help="traget node", action='store', nargs='?', default=1,
                            type=int)    
        self.parser.add_argument("--plot_cluster", help="plot or not clustering", action='store', nargs='?', default=False,
                            type=bool)   
        self.parser.add_argument("--graph", help="graph type", action='store', nargs='?', default='grid',
                            type=str)  
        self.parser.add_argument("--node_size", help="graph type", action='store', nargs='?', default=25,
                            type=int)
        self.parser.add_argument("--Size", help="data size", action='store', nargs='?', default=1000,
                            type=int)
        self.parser.add_argument("--beta", help="infection rate", action='store', nargs='?', default=0.1,
                            type=float)
        self.parser.add_argument("--mu", help="recovery rate", action='store', nargs='?', default=0.4,
                            type=float)
        self.parser.add_argument("--dynamics", help="dynamical model", action='store', nargs='?', default='voter',
                            type=str)              
         
    
    
    def parse_args(self):
        return self.parser.parse_args()

    def parse_args_list(self, args_list):
        return self.parser.parse_args(args_list)