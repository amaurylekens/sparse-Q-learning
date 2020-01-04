import argparse
import json
import sys
from multiprocessing import Process, Manager, Lock
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from MDP_agent import MDPAgent
from game import Game

n_actions = 5
actions_map = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1), 4: (0, 0)}


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
    def f_run(run_times, test_games, size_interval,
              n_episode, line_to_up, run, lock, verbose):

        n_interval = int(n_episode / size_interval)

        Q_table = MDPAgent.get_init_Q_t(n_actions * n_actions, ncol, nrow)

        # game used to run episodes
        learn_game = Game(grid[0], grid[1])

        times = []  # store capture time for each tests in a run
        time = make_capture_test(Q_table, test_games)
        times.append(time)

        for i in range(n_interval):

            Q_table = run_episodes(learn_game, Q_table, size_interval)
            time = make_capture_test(Q_table, test_games)
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
        plt.savefig('images/plots/MDP_{}_{}_grid.png'.format(nrow, ncol))

        data = {"avg": avg, "episode": episode}
        with open('json/MDP_{}_{}_grid.json'.format(nrow, ncol), 'w') as outfile:
            json.dump(data, outfile)


def run_episodes(game: Game, Q_table, n_episode):
    # create predators and prey
    predator = MDPAgent(Q_table)

    # learning parameters
    gamma = 0.9
    epsilon = 0.2
    alpha = 0.3

    for episode in range(n_episode):
        # alpha = 1000/(1000+episode+offset)

        game.reset(random_state=True)

        capture = False
        while not capture:
            # get the current state
            state = state_to_string(game.state)

            # compute the action of the predators
            j_action = predator.get_action_choice(state, epsilon)

            # play the actions and get the reward and the next state
            next_state, reward, capture = game.play(j_action)
            next_state = state_to_string(next_state)

            # make the q-table update
            global_reward = sum(reward.values())
            predator.make_q_update(global_reward, state, j_action, next_state, alpha, gamma)

    return predator.Q_t


def make_capture_test(Q_table, test_games):
    agent = MDPAgent(Q_table)

    capture_times = []
    for game in test_games:
        game.reset()
        capture = False
        while not capture:
            state = state_to_string(game.state)

            j_action = agent.get_action_choice(state, 0)

            _, _, capture = game.play(j_action)

        capture_times.append(game.round)

    return mean(capture_times)


def state_to_string(state):
    return "(" + str(state[0]) + ", " + str(state[1]) + ")"


if __name__ == '__main__':
    Main()
