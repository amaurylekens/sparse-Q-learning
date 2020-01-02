import argparse
import sys
import json
import copy

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

        

def learn_mode(n_episode, grid, directory, verbose=False):

    # make n episodes
    ncol, nrow = grid
    Q_table_1, Q_table_2 = run_episodes(n_episode, grid, verbose)
    # save Q tables
    file_name = "IL_{}_{}_grid".format(ncol, nrow)
    save_tables(Q_table_1, Q_table_2, directory, file_name)
    
    
    
def run_episodes(n_episode, grid, verbose=False, offset=0):

    ncol, nrow = grid
    # create predators and prey
    predators = [ILAgent(0, n_actions[0], ncol, nrow), ILAgent(1, n_actions[1], ncol, nrow)]
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

            # compute and make the rho update
            rules_id = []
            for i, predator in enumerate(predators):
                predator.make_q_update(reward[i], state, j_action[i], next_state, alpha, gamma)

            # move the prey on a free neighbor cell
            free_cells = game.get_free_neighbor_cells()
            action = prey.get_action_choice(free_cells)
            game.play_prey(action)

            round_game += 1

    return predators[0].Q_t, predators[1].Q_t



def save_tables(Q_table_1, Q_table_2, directory, name):

        """
        saves the Q_tables in json format

        :param directory: directory where to save
        :param name: save file name
        """

        data = dict()
        data[0] = Q_table_1
        data[1] = Q_table_2
        path = "{}/{}.json".format(directory, name)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id

def state_to_string(state):
    return "("+str(state[0])+", "+str(state[1])+")"

if __name__ == '__main__':
    Main()