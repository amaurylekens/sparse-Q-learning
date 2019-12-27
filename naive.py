import numpy as np

class Naive():
    def __init__(self, n_actions, rules):

        self.rules = rules
        self.n_actions = n_actions

    def compute_best_j_action(self):

        all_actions = [{0:i, 1:j} for i in range(5) for j in range(5)] 

        counts = []
        for j_action in all_actions:
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

        max_index = [i for i, j in enumerate(counts)
                        if j == max(counts)]

        # choose one of the max-index with uniform distribution
        p = [1/len(max_index) for i in range(len(max_index))]
        best_a = all_actions[np.random.choice(max_index, p=p)]

        return best_a