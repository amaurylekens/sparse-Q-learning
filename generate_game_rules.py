from typing import Tuple

from actions import Actions
from rules import Rules, Rule


def generate_game_rules(grid_size: int) -> Rules:
    """
    generate rules for the prey-predators game
    """
    rules = Rules()

    # two predators
    predator_ids = [0, 1]

    # generate all possible joint states
    all_joint_states = [{0: (i, j), 1: (k, l)}
                        for i in range(grid_size)
                        for j in range(grid_size)
                        for k in range(grid_size)
                        for l in range(grid_size)]

    # generate all combinations of actions
    all_actions = [{0: i, 1: j} for i in Actions.actions for j in Actions.actions]

    # coordination states
    coordination_states = list()

    # collective rules (the 2 predators are close to the prey)
    for state in all_joint_states:
        if (state[0] != state[1] and
                0 < dist(state[1], (0, 0), grid_size) <= 2 and
                0 < dist(state[0], (0, 0), grid_size) <= 2):
            coordination_states.append(state)
            for action in all_actions:
                rule = Rule(state, action, 75)
                rules.add_rule(rule)

    # collective rules (the two predators are close to each other)
    for state in all_joint_states:
        if (0 < dist(state[0], state[1], grid_size) <= 2 and
                dist(state[1], (0, 0), grid_size) > 0 and
                dist(state[0], (0, 0), grid_size) > 0):

            if state not in coordination_states:
                coordination_states.append(state)
                for action in all_actions:
                    rule = Rule(state, action, 75)
                    rules.add_rule(rule)

    # all individual states
    all_states = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
    # individual rules
    for predator in predator_ids:
        for state in all_states:
            for action in Actions.actions:
                rule = Rule({predator: state}, {predator: action}, 75)
                rules.add_rule(rule)

    return rules


def dist(state_1: Tuple[int, int], state_2: Tuple[int, int], grid_size: int) -> int:
    """
    compute Manhattan distance between states
    """
    dist_0 = min(abs(state_1[0] - state_2[0]), grid_size - abs(state_1[0] - state_2[0]))
    dist_1 = min(abs(state_1[1] - state_2[1]), grid_size - abs(state_1[1] - state_2[1]))
    return dist_1 + dist_0