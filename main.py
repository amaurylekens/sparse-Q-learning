#!/anaconda3/bin python3

import os
import sys
import copy
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from coordination_graph import CoordinationGraph
from rules_generator import rules_generator
from agent import Agent
from game import Game
from prey import Prey

n_actions = {0:5, 1:5}
actions_map = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}


class Main(object):

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
        file = "{}_{}_grid.json".format(args.g, args.g)
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

    # create a specific context graph and add rules
    graph = CoordinationGraph(n_actions)
    rules = rules_generator(grid)
    for rule in rules:
        graph.add_rule(rule)

    # make n episodes
    graph = run_episodes(n_episode, grid, graph, verbose)

    file_name = "{}_{}_grid".format(grid, grid)
    graph.save_rules(directory, file_name)


def play_mode(grid, path):

    ncol, nrow = grid

    # create a game
    game = Game({0:(3,0), 1:(0,3)}, ncol, nrow)
    game.print()

    # create a specific context graph and load rules
    n_actions = {0:5, 1:5}
    graph = CoordinationGraph(n_actions)
    graph.load_rules(path)

    # create predators and prey
    predators = [Agent(0, graph, n_actions[0]), Agent(1, graph, n_actions[1])]
    prey = Prey(5)

    while True:
        # get the current state
        state = copy.copy(game.predators)

        # compute the action of the predators
        j_action = dict()
        for i, predator in enumerate(predators):
            j_action[i] = actions_map[predator.get_action_choice(state, 0.2)]

        # play a round
        game.play(j_action)

        # print grid
        game.print()

        choice = ""

        while choice != "s" and choice != "n":
            choice = input("n -> next round, s -> stop : ")
            print(choice)

        if choice == "s":
            break

def test_mode(n_episode, n_run, grid, verbose=False, size_interval=500):
    
    n_interval = int(n_episode/size_interval)

    # generate 100 random initial game states
    ncol, nrow = grid
    all_states = [{0: (i, j), 1: (k, l)}
                  for i in range(ncol)
                  for j in range(nrow)
                  for k in range(ncol)
                  for l in range(nrow)]
    indexes = [random.randint(0, (len(all_states)-1)) for i in range(100)]
    initial_states = [all_states[i] for i in indexes]

    run_times = []  # store the list of ratios for each run
    for run in range(n_run):

        print("run : {}".format(run))
        # create a specific context graph and add rules
        n_actions = {0:5, 1:5}
        rules = rules_generator(grid)
        graph = CoordinationGraph(n_actions)
        for rule in rules:
            graph.add_rule(rule)

        times = []  # store capture time for each tests in a run 
        for i in range(n_interval):
            graph = run_episodes(size_interval, grid, graph, verbose, offset=size_interval*i)
            print("time to test !")
            time = make_capture_test(graph, initial_states, grid)
            times.append(time)

            if verbose:
                print("Test results :")
                print("mean capture time : {}".format(time))

        run_times.append(ratios)

    # average the results over the runs
    avg = [float(sum(col))/len(col) for col in zip(*run_times)]
    episode = np.arange(0, n_episode, interval).tolist()

    plt.plot(episode, avg)
    plt.xlabel("learning episode")
    plt.ylabel("capture/episode")
    plt.title("Evolution of cooperation")
    plt.show()
    plt.savefig('test.png')


def run_episodes(n_episode, grid, graph, verbose=False, offset=0):

    ncol, nrow = grid

    # create predators and prey
    predators = [Agent(0, graph, n_actions[0]), Agent(1, graph, n_actions[1])]
    prey = Prey(5)

    # learning parameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.2

    for episode in range(n_episode):

        if verbose:
            print("episode : {}".format(episode + offset))

        # create a game
        game = Game({0:(1,0), 1:(0,1)}, ncol, nrow)
        game.random_position()

        capture = False
        round_game = 1
        print("round : {}".format(round_game))
        while not capture:
            # get the current state
            state = copy.copy(game.states)

            # compute the action of the predators
            j_action = dict()
            for i, predator in enumerate(predators):
                j_action[i] = actions_map[predator.get_action_choice(state, epsilon)]

            # play the actions and get the reward and the next state
            next_state, reward, capture = game.play(j_action)
            j_action = {id:inv_map(action) for id, action in j_action.items()}

            # compute the best joint action of the next state
            next_j_action = graph.compute_joint_action(next_state)

            # compute and make the rho update
            rules_id = []
            for i, predator in enumerate(predators):
                predator.compute_rho_update(reward[i], state, j_action,
                                            next_state, next_j_action, alpha, gamma) 
            for predator in predators:
                predator.make_rho_update()

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            round_game += 1

            # update printed line
            if verbose:
                sys.stdout.write('\x1b[1A')
                sys.stdout.write('\x1b[2K')
                print("round : {}".format(round_game))

        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

    print("episode : {} -> stop".format(episode + offset))

    return graph


def make_capture_test(graph, initial_states, grid):

    ncol, nrow = grid
    capture_times = []
    agents = [Agent(0, graph, n_actions[0]), Agent(1, graph, n_actions[1])]
    prey = Prey(5)

    # test 5 times for all random initial states
    test_count = 1
    for i, initial_state in enumerate(initial_states):
        for j in range(5):
            game = Game(initial_state, ncol, nrow)

            print("test {}/{}".format(test_count, 5*len(initial_states)))
            capture = False
            capture_time = 0
            while not capture:
                state = copy.copy(game.states)

                j_action = dict()
                for i, agent in enumerate(agents):
                   j_action[i] = actions_map[agent.get_action_choice(state, 0.2)]

                _, _, capture = game.play(j_action)

                # move the prey on a free neighbor cell
                free_cells = game.get_free_neighbor_cells()
                action = prey.get_action_choice(free_cells)
                game.play_prey(action)
                capture_time += 1
            capture_times.append(capture_time)

            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            test_count += 1

    mean_capture_time = mean(capture_times)
    return mean_capture_time


def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id


def nth_root(a, n):
   root = a ** (1/n)
   return root


if __name__ == '__main__':
    Main()