from naive_best_joint_action import NaiveBestJointAction


class CoordinationGraph():
    def __init__(self, n_actions):

        """
        object that represents the specific context coordination graphe

        :param  n_actions: dict with the number of actions for each agent
        """

        self.rules = dict()
        self.n_actions = n_actions

    def add_rule(self, rule):

        """
        add a value rules to the CG

        :param rule: rule to add (dict)
        """

        self.rules[len(self.rules)] = rule

    def compute_joint_action(self, current_state):

        """
        compute the best joint action with the current state

        :param current_state: state of the game
        """

        # select rule which corresponds to the current state
        current_rules = []
        for rule in self.rules.values():
            if rule["state"] == current_state:
                current_rules.append(rule)

        naive = NaiveBestJointAction(self.n_actions, current_rules)

        j_action = naive.compute_best_j_action()

        return j_action

    def get_rules_with_agent(self, agent_id, state, j_action=-1):

        """
        get all the rules where the agent is involved in a particular state

        :param agent_id: id of the agent
        :param state: the particular state
        :param j_action: the j_action that the rules checks
        :return: the corresponding rules
        """

        sel_rules = []
        for rule in self.rules.values():
            states_valided = True
            for agent_id_r, rule_state in rule["state"].items():
                if state[agent_id_r] != rule_state:
                    states_valided = False
                    break
            if states_valided:
                if j_action != -1:
                    actions_valided = True
                    for agent_id_a, action in rule["actions"].items():
                        if j_action[agent_id_a] != action:
                            actions_valided = False
                            break
                    if actions_valided:
                        sel_rules.append(rule)
                else:
                    sel_rules.append(rule)

        # filter rules where the agent is involved
        temp = []
        for rule in sel_rules:
            if agent_id in list(rule["actions"].keys()):
                temp.append(rule)

        sel_rules = temp

        # remove the individual rules if it is coordonate state
        # and inversely
        temp = []
        coordinate = self.is_coordinate_state(agent_id, state)
        for rule in sel_rules:
            if coordinate:
                if len(rule["actions"]) > 1:
                    temp.append(rule)
            else:
                if len(rule["actions"]) == 1:
                    temp.append(rule)
        sel_rules = temp

        return sel_rules

    def is_coordinate_state(self, agent_id, state):

        """
        say if it's coordinated status to the agent requesting it.

        :param agent_id: the requesting agent
        :param state: the state of the game 
        :return: true if it is a coordinate state
        """
        
        coordinate = False
        for rule in self.rules.values():
            if (agent_id in rule["actions"].keys() and
                rule["state"] == state and len(rule["actions"]) > 1):

                coordinate = True

        return coordinate
