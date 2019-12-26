from elimination_algorithm import EliminationAlgorithm


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

        # retrieve the ids of the agents involved
        current_agent_ids = []
        for current_rule in current_rules:
            for agent_id in current_rule["actions"].keys():
                if agent_id not in current_agent_ids:
                    current_agent_ids.append(agent_id)

        # select number of actions for each agent involved
        n_actions = dict()
        for agent_id, n_action in self.n_actions.items():
            if agent_id in current_agent_ids:
                n_actions[agent_id] = n_action

        # elimination to compute the best joint action
        current_agent_ids = sorted(current_agent_ids, reverse=True)
        elimination = EliminationAlgorithm(n_actions, current_rules,
                                           current_agent_ids)
        elimination.forward()
        j_action = elimination.backward()

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
            if (agent_id in rule["actions"].keys() and
                    rule["state"] == state):
                if j_action != -1:
                    rule_valided = True
                    for agent_id, action in rule["actions"].items():
                        if j_action[agent_id] != action:
                            rule_valided = False
                            break
                    if rule_valided:
                        sel_rules.append(rule)
                else:
                    sel_rules.append(rule)

        return sel_rules
