import copy
import numpy as np
import matplotlib.pyplot as plt

from coordination_graph import CoordinationGraph
from rules_generator import rules_generator
from agent import Agent
from game import Game
from prey import Prey

actions_map = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id

def nth_root(a, n):
   root = a ** (1/n) 
   return root

def make_test(n_steps, agents):
    capture_count = 0
    game = Game({0:(3,0), 1:(0,3)}, 4, 4)

    for step in range(n_steps):

        state = copy.copy(game.states)

        j_action = dict()
        for i, agent in enumerate(agents):
           j_action[i] = actions_map[agent.get_action_choice(state, 0.2)]

        next_state, reward, capture = game.play(j_action)
        
        if capture:
            capture_count += 1
     
    if capture_count == 0:
        return -1
    else:
        return (n_steps/capture_count)


# create a game
game = Game({0:(3,0), 1:(0,3)}, 4, 4)

# create a specific context graph and add rules
n_actions = {0:5, 1:5}
graph = CoordinationGraph(n_actions)
rules = rules_generator()
for rule in rules:
    graph.add_rule(rule)

graph.load_rules("rules_files/4_4_grid.json")

# create predators and prey
predators = [Agent(0, graph, n_actions[0]), Agent(1, graph, n_actions[1])]
prey = Prey(5)

steps = 1000
alpha = 0.3
gamma = 0.9
runs = 1


runs_capture_counts = []

for run in range(runs):

    capture_counts = []

    for step in range(1, steps):
        epsilon = 1/nth_root(step, 15)
        
        # get the current state
        state = copy.copy(game.states)

        # compute the action of the predators
        j_action = dict()
        for i, predator in enumerate(predators):
            j_action[i] = actions_map[predator.get_action_choice(state, epsilon)]

        # play the actions and get the reward and the next state
        next_state, reward, found = game.play(j_action)
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
 
        if ((step-1) % 200) == 0:
            print("run : {}, step : {}".format(run, step))
            capture_count = make_test(1000, predators)
            capture_counts.append(capture_count)
            print("steps to capture : {}".format(capture_count))
            print()


    runs_capture_counts.append(capture_counts)

avg = [float(sum(col))/len(col) for col in zip(*runs_capture_counts)]
episode = np.arange(0, steps, 200).tolist()

plt.plot(episode, avg);
plt.show()
plt.savefig('temp.png')


