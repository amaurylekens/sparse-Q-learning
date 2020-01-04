import argparse
import json
import sys
from multiprocessing import Process, Manager, Lock
from statistics import mean
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from game import Game
from generate_game_rules import generate_game_rules
from rules import Rules


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
        parser.add_argument('directory', help="directory of the rules file")
        parser.add_argument('name', help="name of the rules file")
        parser.add_argument('-g', help="grid size", default=4, type=int)

        args = parser.parse_args(sys.argv[2:])
        print('Running play mode, grid={}, directory={} name={}'.format(args.g, args.directory, args.name))

        # run the play mode with the arguments
        grid = (args.g, args.g)

        play_mode(grid, args.directory, args.name)

    def test(self):
        parser = argparse.ArgumentParser(
            description='Test the performance of the learning')
        parser.add_argument('-e', help="number of episode", default=100000, type=int)
        parser.add_argument('-r', help="number of run", default=25, type=int)
        parser.add_argument('-g', help="grid size", default=4, type=int)
        parser.add_argument('-v', '--verbose', action='store_true')

        args = parser.parse_args(sys.argv[2:])
        print('Running test mode, grid={}, run={}, episode={}'.format(args.g, args.r, args.e))

        # run the test mode with the arguments
        grid = (args.g, args.g)
        n_episode = args.e
        n_run = args.r
        verbose = args.verbose
        test_mode(n_episode, n_run, grid, verbose)


def learn_mode(n_episode, grid, directory):
    nrow, ncol = grid
    # create a game
    game = Game(nrow, ncol)

    # create a specific context graph/rules
    rules = generate_game_rules(ncol)

    # create predators
    predators = [Agent(0, rules), Agent(1, rules)]

    # run n episodes
    run_episodes(n_episode, game, rules, predators)

    ncol, nrow = grid
    file_name = "{}_{}_grid".format(ncol, nrow)
    rules.save_rules(directory=directory, name=file_name)


def play_mode(grid, directory, file_name):
    ncol, nrow = grid

    # create a game
    game = Game(ncol, nrow)

    # create a specific context graph and load rules
    rules = Rules()
    rules.load_rules(directory=directory, name=file_name)

    # create predators
    predators = [Agent(0, rules), Agent(1, rules)]

    capture = False

    while not capture:
        state = game.state

        # compute the action of the predators
        j_action = dict()
        for predator in predators:
            j_action[predator.pred_id] = predator.get_action_choice(state, 0.)

        # play the actions and get the reward and the next state
        _, _, capture = game.play(j_action)

        # print grid
        game.print()

        choice = ""

        while choice != "s" and choice != "n":
            choice = input("n -> next episode, s -> stop : ")
            print(choice)

        if choice == "s":
            break


def test_mode(n_episode: int, n_run: int, grid: Tuple[int, int], verbose=False, size_interval: int = 500):
    def f_run(run_times, test_games, size_interval,
              n_episode, line_to_up, run, lock, verbose):

        # create a specific context graph/rules
        rules = generate_game_rules(grid[0])

        # create predator agents
        predators = [Agent(0, rules), Agent(1, rules)]

        # game used to run episodes
        learn_game = Game(grid[0], grid[1])

        n_interval = int(n_episode / size_interval)

        times = []  # store capture time for each tests in a run

        time = make_capture_test(predators, test_games)
        times.append(time)

        for i in range(n_interval):

            run_episodes(size_interval, learn_game, rules, predators)
            time = make_capture_test(predators, test_games)
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
        plt.savefig('images/plots/{}_{}_grid.png'.format(nrow, ncol))

        data = {"avg": avg, "episode": episode}
        with open('json/{}_{}_grid.json'.format(nrow, ncol), 'w') as outfile:
            json.dump(data, outfile)


def run_episodes(n_episode: int, game: Game, rules: Rules, predators: List[Agent]):
    # learning parameters
    gamma = 0.9
    epsilon = 0.2
    alpha = 0.3

    for episode in range(n_episode):
        # reset game to a random initial state
        game.reset(random_state=True)

        capture = False
        while not capture:
            state = game.state

            # compute the action of the predators
            j_action = dict()
            for predator in predators:
                j_action[predator.pred_id] = predator.get_action_choice(state, epsilon)

            # play the actions and get the reward and the next state
            next_state, rewards, capture = game.play(j_action)

            q_values = {predator.pred_id: predator.q_value(state) for predator in predators}

            if not capture:
                future_rewards = {predator.pred_id: predator.q_value(next_state) for predator in predators}
            else:
                future_rewards = {predator.pred_id: 0 for predator in predators}

            rules.update_rule_values(state, j_action, rewards, q_values, future_rewards, alpha, gamma)


def make_capture_test(predators: List[Agent], test_games: List[Game]):
    capture_times = []
    for game in test_games:
        game.reset()
        capture = False
        while not capture:
            state = game.state

            # compute the action of the predators
            j_action = dict()
            for predator in predators:
                j_action[predator.pred_id] = predator.get_action_choice(state, 0.)

            # play the actions and get the reward and the next state
            _, _, capture = game.play(j_action)

        capture_times.append(game.round)

    mean_capture_time = mean(capture_times)

    return mean_capture_time


if __name__ == '__main__':
    Main()
