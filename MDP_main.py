import argparse
import os
import sys
import json
import copy
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random
from multiprocessing import Process, Manager, Lock

from game import Game
from prey import Prey
from MDP_agent import MDPAgent

n_actions = 5
actions_map = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

class Main:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Sparse q-learning',
            usage='''main <command> [<args>]

The commands are:
test      Test the performance of the learning
''')
        parser.add_argument('mode', help='mode to run')
        
        # check the mode
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.mode):
            print('Unrecognized mode')
            parser.print_help()
            exit(1)
        # invoke method with same name
        getattr(self, args.mode)()

    def test(self):
        parser = argparse.ArgumentParser(
            description='Test the performance of the learning')
        parser.add_argument('-e', help="number of episode", default=100000, type=int)
        parser.add_argument('-r', help="number of run", default=25, type=int)
        parser.add_argument('-g', help="grid size", default=4, type=int)
        parser.add_argument('-v', '--verbose', action='store_true')
        
        args = parser.parse_args(sys.argv[2:])
        print('Running test mode, grid={}, run={}, episode{}'.format(args.g, 
                                                                     args.r,
                                                                     args.e))

        # run the test mode with the arguments
        grid = (args.g, args.g)
        n_episode = args.e 
        n_run = args.r
        verbose = args.verbose
        test_mode(n_episode, n_run, grid, verbose)

def test_mode(n_episode, n_run, grid, verbose=False, size_interval=500):
    
    def f_run(run_times, initial_states, size_interval, 
              n_episode, line_to_up, run, lock, verbose):
        
    
        n_interval = int(n_episode/size_interval)

        Q_table = MDPAgent.get_init_Q_t(n_actions*n_actions, ncol, nrow)

        times = []  # store capture time for each tests in a run
        time = make_capture_test(Q_table, initial_states, grid, verbose)
        times.append(time) 

        for i in range(n_interval):

            Q_table = run_episodes(Q_table, size_interval, grid, verbose, offset=size_interval*i)
            time = make_capture_test(Q_table, initial_states, grid, verbose)
            times.append(time)

            if verbose:
                lock.acquire()
                for line in range(line_to_up):
                    sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                sys.stdout.write("run {} : {} %".format((run+1), ((i+1)/n_interval)*100))
                for line in range(line_to_up):
                    print("\r")
                lock.release()

        run_times.append(times)


    # generate 100 random initial game states
    ncol, nrow = grid
    all_states = [{0: (i, j), 1: (k, l)}
                  for i in range(ncol)
                  for j in range(nrow)
                  for k in range(ncol)
                  for l in range(nrow)
                  if not ((i==k and j==l) or (i==0 and j==0) or (k==0 and l==0))]
    
    indexes = [random.randint(0, (len(all_states)-1)) for i in range(100)]
    initial_states = [all_states[i] for i in indexes]


    with Manager() as manager:
        run_times= manager.list()  # <-- can be shared between processes.
        processes = []
        N_PROCESS = 8
        lock = Lock()
        for run in range(n_run):
            p = Process(target=f_run, args=(run_times, initial_states, 
                                            size_interval, n_episode, 
                                            (n_run-run), run, lock, verbose))
            if verbose: 
                print("run {} : {} %".format((run+1), 0))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


        # average the results over the runs
        avg = [float(sum(col))/len(col) for col in zip(*run_times)]
        episode = np.arange(0, n_episode + size_interval, size_interval).tolist()


        plt.plot(episode, avg);
        plt.xlabel("learning episode")
        plt.ylabel("capture/episode")
        plt.title("Evolution of cooperation")
        plt.savefig('images/plots/MDP_{}_{}_grid.png'.format(nrow, ncol))

        data = {"avg": avg, "episode": episode}
        with open('json/MDP_{}_{}_grid.json'.format(nrow, ncol), 'w') as outfile:
             json.dump(data, outfile)

def run_episodes(Q_table, n_episode, grid, verbose=False, offset=0):

    ncol, nrow = grid
    # create predators and prey
    predator = MDPAgent(n_actions, ncol, nrow, Q_table)
    prey = Prey(5)

    # learning parameters
    gamma = 0.9
    epsilon = 0.2
    alpha = 0.3

    for episode in range(n_episode):

        #alpha = 1000/(1000+episode+offset)

        # create a game
        game = Game({0:(1,0), 1:(0,1)}, ncol, nrow)
        game.random_position()

        capture = False
        round_game = 1

        while not capture:
            # get the current state
            state = state_to_string(copy.copy(game.states))

            # compute the action of the predators
            j_action = dict()
            joint_action = predator.get_action_choice(state, epsilon)
            j_action[0] = actions_map[joint_action[0]]
            j_action[1] = actions_map[joint_action[1]]

            # play the actions and get the reward and the next state
            next_state, reward, capture = game.play(j_action)
            next_state = state_to_string(next_state)
            j_action = {id:inv_map(action) for id, action in j_action.items()}

            # make the q-table update
            global_reward = reward[0] + reward[1]
            predator.make_q_update(global_reward, state, j_action, next_state, alpha, gamma)

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            round_game += 1

    return predator.Q_t

def make_capture_test(Q_table, initial_states, grid, verbose=False):
    
    ncol, nrow = grid
    capture_times = []
    agent = MDPAgent(n_actions, ncol, nrow, Q_table)
    prey = Prey(5)

   
    capture_times = []
    for initial_state in initial_states:
        game = Game(copy.copy(initial_state), ncol, nrow)
        
        capture = False
        capture_time = 0
        while not capture:
            state = state_to_string(copy.copy(game.states))
            j_action = dict()
            joint_action = agent.get_action_choice(state, 0.2) # TODO : Why not eps = 0
            j_action[0] = actions_map[joint_action[0]]
            j_action[1] = actions_map[joint_action[1]]
            
            _, _, capture = game.play(j_action)

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            capture_time += 1

        capture_times.append(capture_time)

    mean_capture_time = mean(capture_times)
    
    return mean_capture_time

def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id

def state_to_string(state):
    return "("+str(state[0])+", "+str(state[1])+")"

if __name__ == '__main__':
    Main()