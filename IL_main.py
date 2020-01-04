import argparse
import json
import os
import sys
from multiprocessing import Process, Manager, Lock
from statistics import mean
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from IL_agent import ILAgent
from game import Game


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
                            help="directory to store the q-tables file")
        parser.add_argument('-e', help="number of episode", default=100000, type=int)
        parser.add_argument('-g', help="grid size", default=4, type=int)

        args = parser.parse_args(sys.argv[2:])
        print('Running learn mode, episode={}, grid={}'.format(args.e, args.g))

        # run the learn mode with the arguments
        n_episode = args.e
        grid = (args.g, args.g)
        directory = args.directory
        learn_mode(n_episode, grid, directory)

    def play(self):
        # manage the arguments
        parser = argparse.ArgumentParser(
            description='Play the game with a learned policy')
        parser.add_argument('directory', help="directory of the q-tables file")
        parser.add_argument('-g', help="grid size", default=4, type=int)

        args = parser.parse_args(sys.argv[2:])
        print('Running play mode, grid={}, directory={}'.format(args.g, args.directory))

        # run the play mode with the arguments
        grid = (args.g, args.g)
        file = "IL_{}_{}_grid.json".format(args.g, args.g)
        path = os.path.join(args.directory, file)

        # check if there is a good q-tables file, if there is let's play
        if os.path.exists(path):
            play_mode(grid, path)
        else:
            print("no q-tables file in this directory for this grid size")

    def test(self):
        parser = argparse.ArgumentParser(
            description='Test the performance of the learning')
        parser.add_argument('-e', help="number of episode", default=100000, type=int)
        parser.add_argument('-r', help="number of run", default=25, type=int)
        parser.add_argument('-g', help="grid size", default=4, type=int)
        parser.add_argument('-v', '--verbose', action='store_true')

        args = parser.parse_args(sys.argv[2:])
        print('Running test mode, grid={}, run={}, episode={}'.format(args.g,
                                                                      args.r,
                                                                      args.e))

        # run the test mode with the arguments
        grid = (args.g, args.g)
        n_episode = args.e
        n_run = args.r
        verbose = args.verbose
        test_mode(n_episode, n_run, grid, verbose)


def learn_mode(n_episode, grid, directory):
    # make n episodes
    ncol, nrow = grid
    Q_table_0, Q_table_1 = run_episodes(Game(ncol, nrow),
                                        ILAgent.get_init_Q_t(ncol, nrow),
                                        ILAgent.get_init_Q_t(ncol, nrow),
                                        n_episode)
    # save Q tables
    file_name = "IL_{}_{}_grid".format(ncol, nrow)
    save_tables(Q_table_0, Q_table_1, directory, file_name)


def play_mode(grid, path):
    ncol, nrow = grid

    # create a game
    game = Game(ncol, nrow)
    game.print()

    # load Q_tables
    Q_table_0, Q_table_1 = load_tables(path)

    # create predators and prey
    predators = [ILAgent(0, Q_table_0), ILAgent(1, Q_table_1)]

    capture = False
    while not capture:
        # get the current state
        state = state_to_string(game.state)

        # compute the action of the predators
        j_action = dict()
        for i, predator in enumerate(predators):
            j_action[i] = predator.get_action_choice(state, 0)

        # play a episode
        _, _, capture = game.play(j_action)

        # print grid
        game.print()

        choice = ""

        while choice != "s" and choice != "n":
            choice = input("n -> next episode, s -> stop : ")
            print(choice)

        if choice == "s":
            break


def test_mode(n_episode, n_run, grid, verbose=False, size_interval=500):
    def f_run(run_times, test_games, size_interval,
              n_episode, line_to_up, run, lock, verbose):

        n_interval = int(n_episode / size_interval)

        Q_table_0 = ILAgent.get_init_Q_t(ncol, nrow)
        Q_table_1 = ILAgent.get_init_Q_t(ncol, nrow)

        # game used to run episodes
        learn_game = Game(grid[0], grid[1])

        times = []  # store capture time for each tests in a run
        time = make_capture_test(Q_table_0, Q_table_1, test_games)
        times.append(time)

        for i in range(n_interval):

            Q_table_0, Q_table_1 = run_episodes(learn_game, Q_table_0, Q_table_1, size_interval)
            time = make_capture_test(Q_table_0, Q_table_1, test_games)
            times.append(time)

            if verbose:
                lock.acquire()
                for line in range(line_to_up):
                    sys.stdout.write("\033[F")
                sys.stdout.write("\033[K")
                sys.stdout.write("run {} : {} %".format((run + 1), ((i + 1) / n_interval) * 100))
                for line in range(line_to_up):
                    print("\r")
                lock.release()

        run_times.append(times)

    # generate 100 random initial game states
    ncol, nrow = grid
    test_games = [Game(nrow, ncol) for _ in range(100)]

    with Manager() as manager:
        run_times = manager.list()  # <-- can be shared between processes.
        processes = []
        N_PROCESS = 8
        lock = Lock()
        for run in range(n_run):
            p = Process(target=f_run, args=(run_times, test_games,
                                            size_interval, n_episode,
                                            (n_run - run), run, lock, verbose))
            if verbose:
                print("run {} : {} %".format((run + 1), 0))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # average the results over the runs
        avg = [float(sum(col)) / len(col) for col in zip(*run_times)]
        episode = np.arange(0, n_episode + size_interval, size_interval).tolist()

        plt.plot(episode, avg)
        plt.xlabel("learning episode")
        plt.ylabel("capture/episode")
        plt.title("Evolution of cooperation")
        plt.savefig('images/plots/IL_{}_{}_grid.png'.format(nrow, ncol))

        data = {"avg": avg, "episode": episode}
        with open('json/IL_{}_{}_grid.json'.format(nrow, ncol), 'w') as outfile:
            json.dump(data, outfile)


def run_episodes(game: Game, Q_table_0, Q_table_1, n_episode):
    # create predators and prey
    predators = [ILAgent(0, Q_table_0), ILAgent(1, Q_table_1)]

    # learning parameters
    gamma = 0.9
    epsilon = 0.2
    alpha = 0.3

    for episode in range(n_episode):
        # alpha = 1000/(1000+episode+offset)

        capture = False

        while not capture:
            game.reset(random_state=True)
            # get the current state
            state = state_to_string(game.state)

            # compute the action of the predators
            j_action = dict()
            for i, predator in enumerate(predators):
                j_action[i] = predator.get_action_choice(state, epsilon)

            # play the actions and get the reward and the next state
            next_state, reward, capture = game.play(j_action)
            next_state = state_to_string(next_state)

            # make the q-table update
            for i, predator in enumerate(predators):
                predator.make_q_update(reward[i], state, j_action[i], next_state, alpha, gamma)

    return predators[0].Q_t, predators[1].Q_t


def make_capture_test(Q_table_0, Q_table_1, test_games: List[Game]):

    agents = [ILAgent(0, Q_table_0), ILAgent(1, Q_table_1)]

    capture_times = []
    for game in test_games:
        game.reset()
        capture = False
        while not capture:
            state = state_to_string(game.state)
            j_action = dict()
            for i, agent in enumerate(agents):
                j_action[i] = agent.get_action_choice(state, 0)

            _, _, capture = game.play(j_action)

        capture_times.append(game.round)

    return mean(capture_times)


def save_tables(Q_table_0, Q_table_1, directory, name):
    """
    saves the Q-tables in json format

    :param directory: directory where to save
    :param name: save file name
    """

    data = dict()
    data[0] = Q_table_0
    data[1] = Q_table_1
    path = os.path.join(directory, f'{name}.json')
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


def state_to_string(state):
    return "(" + str(state[0]) + ", " + str(state[1]) + ")"


if __name__ == '__main__':
    Main()
