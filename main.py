from coordination_graph import CoordinationGraph
from agent import Agent
from game import Game
from prey import Prey
from rules_generator import rules_generator


actions_map = {0: (1,0), 1:(0,1), 2:(-1,0), 3:(0,-1), 4:(0,0)}

def inv_map(action_to_map):
    for action_id, action in actions_map.items():
        if action_to_map == action:
            return action_id

# create a game
game = Game({0:(5,2), 1:(3,1)})

# create a specific context graph and add rules
n_actions = {0:5, 1:5}
graph = CoordinationGraph(n_actions)
rules = rules_generator()
for rule in rules:
    graph.add_rule(rule)


# create predators and prey
predators = [Agent(0, graph, n_actions[0]), Agent(1, graph, n_actions[1])]
prey = Prey(5)

steps = 100000
alpha = 0.3
gamma = 0.9

# play the game
found_count = 0
for step in range(steps):
    print("step {}".format(step))
    
    # get the current state
    state = game.states

    # compute the action of the predators and the prey
    actions = dict()
    j_action = dict()
    for i, predator in enumerate(predators):
        j_action[i] = actions_map[predator.get_action_choice(state)]
    actions["pred"] = j_action
    actions["prey"] = actions_map[prey.get_action_choice()]

    # play the actions and get the reward and the next state
    next_state, reward, found = game.play(actions)
    if found:
        found_count += 1

    j_action = {id:inv_map(action) for id, action in j_action.items()}

    # compute the best joint action of the next state
    next_j_action = graph.compute_joint_action(next_state)

    # compute and make the rho update
    for i, predator in enumerate(predators):
        predator.compute_rhos_updates(reward[i], state, j_action,
                                      next_state, next_j_action, alpha, gamma)
        #print(predator.rhos_updates)
    
    for predator in predators:
        predator.make_rhos_updates()

    if step%500 == 0:
        print("found : {}".format(found_count))
        found_count = 0

    #game.print()









