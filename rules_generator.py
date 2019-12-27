def rules_generator():

    """
    generate rules for the prey-predators game
    """

    ncol = 4
    nrow = 4

    rules = []  # store the rules
    coor_states = []  # store the coordinate states
    predator_ids = [0, 1]
    gap = 10

    # generate all possible states
    all_states = [{0: (i, j), 1: (k, l)}
                  for i in range(ncol)
                  for j in range(nrow)
                  for k in range(ncol)
                  for l in range(nrow)]
    n_actions = 5  # number of actions

    # generate all combinations of actions
    all_actions = [(i, j) for i in range(n_actions) for j in range(n_actions)]

    # collective rules (the 2 predators are close to the prey)
    for state in all_states:
        if (dist(state[0], (0, 0)) <= 2 and
                dist(state[1], (0, 0)) <= 2 and
                dist(state[1], (0, 0)) > 0 and
                dist(state[0], (0, 0)) > 0 and
                dist(state[0], state[1]) > 0):

            coor_states.append(state)

            for action in all_actions:
                actions = {0: action[0], 1: action[1]}
                rho = 75
                id_ = len(rules)
                rule = {"state": state, "actions": actions,
                        "id": id_, "rho": rho}
                rules.append(rule)

    # collective rules (the two predators are close to each other)
    for state in all_states:
        if (dist(state[0], state[1]) <= 2 and dist(state[0], state[1]) > 0 and
                dist(state[1], (0, 0)) > 0 and dist(state[0], (0, 0)) > 0):

            if state not in coor_states:
                coor_states.append(state)
                for action in all_actions:
                    actions = {0: action[0], 1: action[1]}
                    rho = 75
                    id = len(rules)
                    rule = {"state": state, "actions": actions,
                            "id": id, "rho": rho}
                    rules.append(rule)

    all_states = [(i, j) for i in range(ncol) for j in range(nrow)]
    # individual rules
    for id in predator_ids:
        for state in all_states:
            if (dist(state, (0, 0)) <= gap and
                    dist(state, (0, 0)) > 0):

                for action in range(n_actions):
                    actions = {id: action}
                    rho = 75
                    id_ = len(rules)
                    rule = {"state": {id: state}, "actions": actions,
                            "id": id_, "rho": rho}
                    rules.append(rule)
    return rules


def dist(state_1, state_2):

    """
    compute Manhatan distance between to cell of the grid
    """

    dist_0 = min(abs(state_1[0] - state_2[0]), 10-abs(state_1[0] - state_2[0]))
    dist_1 = min(abs(state_1[1] - state_2[1]), 10-abs(state_1[1] - state_2[1]))

    return dist_1+dist_0
