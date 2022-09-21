from netrd.dynamics import BaseDynamics
import numpy as np
import networkx as nx
import pandas as pd
from graphs import KGraph
from parsers import KParseArgs
import sys

class Generate_Data():

    def __init__(self) -> None:
        pass

    

    def voter(self,args):
        def simulate_voter(args,G, L, noise=None):
            N = args.node_size

            if noise is None:
                noise = 0
            elif noise == 'automatic' or noise == 'auto':
                noise = 1 / N
            elif not isinstance(noise, (int, float)):
                raise ValueError("noise must be a number, 'automatic', or None")

            transitions = nx.to_numpy_array(G)
            transitions = transitions / np.sum(transitions, axis=0)

            TS = np.zeros((N, L))
            TS[:, 0] = [1 if x < 0.5 else 0 for x in np.random.rand(N)]
            indices = np.arange(N)

            for t in range(1, L):
                np.random.shuffle(indices)
                TS[:, t] = TS[:, t - 1]
                for i in indices:
                    TS[i, t] = np.random.choice(TS[:, t], p=transitions[:, i])
                    if np.random.rand() < noise:
                        TS[i, t] = 1 if np.random.rand() < 0.5 else 0
                if np.array_equal(TS.T[t-1],TS.T[t-2]):
                    TS = TS.T
                    TS = TS[:t]
                    break
            return TS[:-1]

        def gen_voter_data(Size,graph):
            data = []
            for i in range(Size):
                new_sample = simulate_voter(args,G = graph,L = 1000,noise = 0.01)
                data.append(new_sample)
                a = np.concatenate(data)
                if a.shape[0]>Size:
                    break
            
   
            return np.concatenate(data)
            
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args)
        TS = gen_voter_data(args.Size,G)
        df = pd.DataFrame(TS)
        df.to_csv('data/'+str(args.graph)+'_'+str(args.node_size)+'_voter_'+ str(args.Size)+'.csv',index = False)
        return print('Data Generated Successfully')


    def sis(self,args):
        def simulate(G, L, num_seeds=1, beta=None, mu=None, dontdie=False):
            H = G.copy()
            N = H.number_of_nodes()
            TS = np.zeros((N, L))
            index_to_node = dict(zip(range(G.order()), list(G.nodes())))

            # sensible defaults for beta and mu
            if not beta:
                avg_k = np.mean(list(dict(H.degree()).values()))
                beta = 1 / avg_k
            if not mu:
                mu = 1 / H.number_of_nodes()

            seeds = np.random.permutation(np.concatenate([np.repeat(1, num_seeds), np.repeat(0, N - num_seeds)]))
            TS[:, 0] = seeds
            infected_attr = {index_to_node[i]: s for i, s in enumerate(seeds)}
            nx.set_node_attributes(H, infected_attr, 'infected')
            nx.set_node_attributes(H, 0, 'next_infected')

            # SIS dynamics
            num_dontdie=0
            for t in range(1, L):
                #print('Test')
                nodes = np.random.permutation(H.nodes)
                for i in nodes:
                    if H.nodes[i]['infected']:
                        neigh = H.neighbors(i)
                        for j in neigh:
                            if np.random.random() < beta:
                                H.nodes[j]['next_infected'] = 1
                        if np.random.random() >= mu:
                            H.nodes[i]['next_infected'] = 1
                infections = nx.get_node_attributes(H, 'infected')
                next_infections = nx.get_node_attributes(H, 'next_infected')
                    #print(infections)
                    #print(next_infections==0)
                if dontdie and all([x==0 for x in next_infections.values()]):
                #print('dontdie')
                    num_dontdie += 1
                    key = list(infections.keys())[list(infections.values()).index(1)]
                    next_infections[key] = 1

                # store SIS dynamics for time t
                TS[:, t] = np.array(list(infections.values()))
                nx.set_node_attributes(H, next_infections, 'infected')
                nx.set_node_attributes(H, 0, 'next_infected')

                # if the epidemic dies off, stop
                if TS[:, t].sum() < 1:
                    break

                if TS.shape[1] < L:
                    TS = np.hstack([TS, np.zeros((N, L - TS.shape[1]))])
            
            return TS
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args)
        TS = simulate(G, args.Size, num_seeds=1, beta=args.beta, mu=args.mu, dontdie=True)
        df = pd.DataFrame(TS.T)
        df.to_csv('data/'+str(args.graph)+'_'+str(args.node_size)+'_sis_'+ str(args.Size)+'.csv',index = False)
        return print('Data Generated Successfully')

    def gol(self,args):
        def game_of_life(state,graph,b,c,d):
            nextstate = np.zeros_like(state)
    
            for i in range(len(state)):
                # determine number of "opinionated" neighbors of node i
                neighbors_iter = graph.neighbors(i)
                num_alive_neighbours  = sum(1 for pred in neighbors_iter if state[pred] == 1)
                num_dead_neighbours = len(graph[i].keys())-num_alive_neighbours
                #print(num_alive_neighbours,num_dead_neighbours)
                if state[i] == 1:
                    if num_alive_neighbours>=b and num_dead_neighbours>=c:
                        nextstate[i] = 1
                elif state[i] == 0:
                    if num_alive_neighbours == d:
                        nextstate[i] = 1
                else:
                    nextstate[i] = 0
            
            return np.array(nextstate)

        def gen_sequence_life_of_game(visited_states,graph,b,c,d,condition):
            NUM_NODES = graph.number_of_nodes()
            n = int(1000)
    
            cand = np.random.randint(2, size=NUM_NODES)
            just_visited = []
            cur_state = cand
            just_visited.append(cand)
            for _ in range(n):
                next_state = game_of_life(cur_state,graph,b,c,d)
                if condition=="any":
                    if len(just_visited)>0 and any([np.array_equal(next_state, jv) for jv in just_visited]): #np.array_equal(just_visited[-1], next_state):
                        just_visited.append(next_state)
                        break
                    else:
                        just_visited.append(next_state)
                        cur_state = next_state
                
                elif condition == "end":
                    if len(just_visited)>0 and np.array_equal(just_visited[-1], next_state): #np.array_equal(just_visited[-1], next_state):
                        just_visited.append(next_state)
                        break
            
                    else:
                        just_visited.append(next_state)
                        cur_state = next_state
            return np.array(just_visited)

        def game_of_gen_data(n,G):
            data = []
            for i in range(n):
                seq_grid = gen_sequence_life_of_game([],G,1,1,1,"any")
                data.append(seq_grid)
                sample = np.concatenate(data)
                if sample.shape[0]>n:
                    break
        
            return np.concatenate(data)
        
        graph_generator = KGraph()
        G = graph_generator.generate_graph(args)
        TS = game_of_gen_data(args.Size,G)
        df = pd.DataFrame(TS)
        df.to_csv('data/'+str(args.graph)+'_'+str(args.node_size)+'_GOL_'+ str(args.Size)+'.csv',index = False)
        return print('Data Generated Successfully')

        

if __name__=='__main__':
    parser = KParseArgs()
    args = parser.parse_args()
    flag = len(sys.argv) == 1
    data_generator = Generate_Data()
    if args.dynamics =='voter':
        data_generator.voter(args)
    elif args.dynamics == 'sis':
        data_generator.sis(args)
    elif args.dynamics == 'gol':
        data_generator.gol(args)



