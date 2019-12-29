import random
import numpy as np


class Agent:
    def __init__(self, id, coord_graph, n_action):

        """
        create a agent with a Boltzmann action selection
        and with sparse Q-learning

        :param agent_id: id of the agent
        :coord_graph: context-specific coordination graphe
        :n_action: number of actions choice
        """

        self.id = id
        self.coord_graph = coord_graph
        self.n_action = n_action
        self.rho_update = {}

    def get_action_choice(self, state, epsilon):

        """
        return the agent's choice of action for a particular state

        :param state: the state of the game
        :param epsilon: e-greedy parameter
        :return: the agent's choice of action
        """

        # e-greedy
        if (random.uniform(0, 1) < epsilon):
            a_t = random.randint(0, self.n_action-1)
        else:
            # recuperate all rules for the current state where the agent in involved
            rules = self.coord_graph.get_rules_with_agent(self.id, state)

            # compute the Q_t for each joint_action 
            j_actions = [{0:i, 1:j} for i in range(self.n_action) for j in range(self.n_action)]
            Qs_t = [0]*len(j_actions)
            for rule in rules:
                for j_action in j_actions:
                    rule_validated = True
                    for agent_id, action in rule["actions"].items():
                        if j_action[agent_id] != action: 
                            rule_validated = False
                            break
                    if rule_validated:
                        index = j_actions.index(j_action)
                        Qs_t[index] += rule["rho"]/len(rule["actions"])

                else:
                    j_actions.append(rule["actions"])
                    Qs_t.append(rule["rho"]/len(rule["actions"]))

            # find index of the max Q-values
            max_index = [i for i, j in enumerate(Qs_t)
                         if j == max(Qs_t)]

            # choose one of the max-index with uniform distribution
            p = [1/len(max_index) for i in range(len(max_index))]
            sel_index = np.random.choice(max_index, p=p)
            a_t = j_actions[sel_index][self.id]

        return a_t
            

    def compute_rho_update(self, reward, state, action,
                           next_state, next_action, alpha, gamma):

        """
        compute the values to be added to each rho in which
        the agent is involved.

        :param reward: reward received for the last action choice
        :param state: the state before the joint action
        :param action: the joint action
        :param next_state: the state resulting of the joint action
        :param next_action: the best joint action for the next state
        :param alpha: learning rate
        :param gamma: discount factor
        """

        # retrieve the rule that correspond to the action-state
        rule = self.coord_graph.get_rules_with_agent(self.id,
                                                     state,
                                                     action)[0]

        # retrieve the rule that correspond to the next action-state
        rule_next = self.coord_graph.get_rules_with_agent(self.id,
                                                          next_state,
                                                          next_action)[0]

        # compute the update and store it
        self.rho_update = dict()
        rule_id = rule["id"]

        update = alpha*reward
        weighted_rho = rule_next["rho"]/len(rule_next["actions"])
        update += alpha*gamma*(weighted_rho)
        update -= alpha*rule["rho"]/len(rule["actions"])

        self.rho_update[rule_id] = update

    def make_rho_update(self):

        """
        make the update of the rhos on the coord_graph
        with the computed update-values
        """

        for rule_id,  update in self.rho_update.items():
            self.coord_graph.rules[rule_id]["rho"] += update
