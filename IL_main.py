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

from IL_agent import ILAgent
from game import Game
from prey import Prey

n_actions = {0:5, 1:5}
actions_map = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

class Main:

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Sparse q-learning',
            usage='''main <command> [<args>]

The commands are:
learn     Let the agents learn a policy during n episodes
play      Play the game with a learned policy
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

    def learn(self):
        # manage the arguments
        parser = argparse.ArgumentParser(
            description='Let the agents learn a policy during n episodes')
        parser.add_argument('directory', 
            help="directory to store the rules file")
        parser.add_argument('-e', help="number of episode", default=100000, type=int)
        parser.add_argument('-g', help="grid size", default=4, type=int)
        parser.add_argument('-v', '--verbose', action='store_true')
        
        args = parser.parse_args(sys.argv[2:])
        print('Running learn mode, episode={}, grid={}'.format(args.e, args.g))

        # run the learn mode with the arguments
        n_episode = args.e
        grid = (args.g, args.g)
        directory = args.directory
        verbose = args.verbose
        learn_mode(n_episode, grid, directory, verbose)

    def play(self):
        # manage the arguments
        parser = argparse.ArgumentParser(
            description='Play the game with a learned policy')
        parser.add_argument('directory', help="directory of the rules file")
        parser.add_argument('-g', help="grid size", default=4, type=int)
        
        args = parser.parse_args(sys.argv[2:])
        print('Running play mode, grid={}, directory={}'.format(args.g, args.directory))

        # run the play mode with the arguments
        grid = (args.g, args.g)
        file = "IL_{}_{}_grid.json".format(args.g, args.g)
        path = "{}/{}".format(args.directory, file)
        
        # check if there is a good rules file, if there is let's play
        if file in os.listdir("./{}".format(args.directory)):
            play_mode(grid, path)
        else:
            print("no rule file in this directory for this grid size")

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

        

def learn_mode(n_episode, grid, directory, verbose=False):
    # make n episodes
    ncol, nrow = grid
    Q_table_0, Q_table_1 = run_episodes(ILAgent.get_init_Q_t(n_actions[0], ncol, nrow), ILAgent.get_init_Q_t(n_actions[1], ncol, nrow), n_episode, grid, verbose)
    # save Q tables
    file_name = "IL_{}_{}_grid".format(ncol, nrow)
    save_tables(Q_table_0, Q_table_1, directory, file_name)
    

def play_mode(grid, path):
    ncol, nrow = grid

    # create a game
    game = Game({0:(3,0), 1:(0,3)}, ncol, nrow)
    game.print()

    # load Q_tables
    Q_table_0, Q_table_1 = load_tables(path)

    # create predators and prey
    predators = [ILAgent(0, n_actions[0], ncol, nrow, Q_table_0), ILAgent(1, n_actions[1], ncol, nrow, Q_table_1)]
    prey = Prey(5)

    while True:
        # get the current state
        state = state_to_string(copy.copy(game.states))

        # compute the action of the predators
        j_action = dict()
        for i, predator in enumerate(predators):
            j_action[i] = actions_map[predator.get_action_choice(state, 0)]

        # play a episode
        game.play(j_action)

        # move the prey on a free neighbor cell
        free_cells = game.get_free_neighbor_cells()
        action = prey.get_action_choice(free_cells)
        game.play_prey(action)

        # print grid
        game.print()

        choice = ""

        while choice != "s" and choice != "n":
            choice = input("n -> next episode, s -> stop : ")
            print(choice)

        if choice == "s":
            break
    
def test_mode(n_episode, n_run, grid, verbose=False, size_interval=500):
    
    def f_run(run_times, initial_states, size_interval, 
              n_episode, line_to_up, run, lock, verbose):
        
        # create a specific context graph and add rules
        n_actions = {0:5, 1:5}
    
        n_interval = int(n_episode/size_interval)

        Q_table_0 = ILAgent.get_init_Q_t(n_actions[0], ncol, nrow)
        Q_table_1 = ILAgent.get_init_Q_t(n_actions[1], ncol, nrow)

        times = []  # store capture time for each tests in a run
        time = make_capture_test(Q_table_0, Q_table_1, initial_states, grid, verbose)
        times.append(time) 

        for i in range(n_interval):

            Q_table_0, Q_table_1 = run_episodes(Q_table_0, Q_table_1, size_interval, grid, verbose, offset=size_interval*i)
            time = make_capture_test(Q_table_0, Q_table_1, initial_states, grid, verbose)
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
        plt.savefig('images/plots/IL_{}_{}_grid.png'.format(nrow, ncol))

        data = {"avg": avg, "episode": episode}
        with open('json/{}_{}_grid.json'.format(nrow, ncol), 'w') as outfile:
             json.dump(data, outfile)

def run_episodes(Q_table_0, Q_table_1, n_episode, grid, verbose=False, offset=0):

    ncol, nrow = grid
    # create predators and prey
    predators = [ILAgent(0, n_actions[0], ncol, nrow, Q_table_0), ILAgent(1, n_actions[1], ncol, nrow, Q_table_1)]
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
            for i, predator in enumerate(predators):
                j_action[i] = actions_map[predator.get_action_choice(state, epsilon)]

            # play the actions and get the reward and the next state
            next_state, reward, capture = game.play(j_action)
            next_state = state_to_string(next_state)
            j_action = {id:inv_map(action) for id, action in j_action.items()}

            # make the q-table update
            for i, predator in enumerate(predators):
                predator.make_q_update(reward[i], state, j_action[i], next_state, alpha, gamma)

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            round_game += 1

    return predators[0].Q_t, predators[1].Q_t

def make_capture_test(Q_table_0, Q_table_1, initial_states, grid, verbose=False):
    
    ncol, nrow = grid
    capture_times = []
    agents = [ILAgent(0, n_actions[0], ncol, nrow, Q_table_0), ILAgent(1, n_actions[1], ncol, nrow, Q_table_1)]
    prey = Prey(5)

   
    capture_times = []
    for initial_state in initial_states:
        game = Game(copy.copy(initial_state), ncol, nrow)
        
        capture = False
        capture_time = 0
        while not capture:
            state = state_to_string(copy.copy(game.states))
            j_action = dict()
            for i, agent in enumerate(agents):
               j_action[i] = actions_map[agent.get_action_choice(state, 0.2)] # TODO : Why not eps = 0

            _, _, capture = game.play(j_action)

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            capture_time += 1

        capture_times.append(capture_time)

    mean_capture_time = mean(capture_times)
    
    return mean_capture_time

def save_tables(Q_table_0, Q_table_1, directory, name):

        """
        saves the Q-tables in json format

        :param directory: directory where to save
        :param name: save file name
        """

        data = dict()
        data[0] = Q_table_0
        data[1] = Q_table_1
        path = "{}/{}.json".format(directory, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def load_tables(path):

        """
        load the Q-tables from a json file

        :param path: path to the json file
        """

        with open(path) as f:
            data = json.load(f)

        Q_table_0 = data["0"]
        Q_table_1 = data["1"]

        return Q_table_0, Q_table_1

def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id

def state_to_string(state):
    return "("+str(state[0])+", "+str(state[1])+")"

if __name__ == '__main__':
    Main()