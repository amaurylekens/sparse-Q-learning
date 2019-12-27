import numpy as np


class NaiveBestJointAction():
    def __init__(self, n_actions, rules):

        """
        Allow to compute the best joint action with a set of rules
        and a set of actions

        :param n_actions: number of actions for each agents
        :param rules: rules used to compute the best joint action
        """

        self.rules = rules
        self.n_actions = n_actions

    def compute_best_j_action(self):

        """
        Allow to compute the best joint action with a set of rules
        and a set of actions
        """

        # all possible joint action
        j_actions = [{0: i, 1: j} for i in range(5) for j in range(5)]

        # compute reward for each joint action
        counts = []
        for j_action in j_actions:
            count = 0
            for rule in self.rules:
                rule_valided = True
                for agent_id, action in rule["actions"].items():
                    if j_action[agent_id] != action:
                        rule_valided = False
                        break
                if rule_valided:
                    count += rule["rho"]
            counts.append(count)

        # return the joint action with the best reward
        max_index = [i for i, j in enumerate(counts)
                     if j == max(counts)]

        # choose one of the max-index with uniform distribution
        p = [1/len(max_index) for i in range(len(max_index))]
        best_j_action = j_actions[np.random.choice(max_index, p=p)]

        return best_j_action
