import random
import numpy as np

from actions import Actions
from rules import Rules


class Agent:
    def __init__(self, pred_id, rules: Rules):

        """
        create a agent with a Boltzmann action selection
        and with sparse Q-learning

        :param pred_id: id of the agent
        :rules: rules
        """

        self.pred_id = pred_id
        self.rules = rules
        self.rho_update = {}

    def get_action_choice(self, state, epsilon):
        """
        return the agent's choice of action for a particular state

        :param state: the state of the game
        :param epsilon: e-greedy parameter
        :return: the agent's choice of action
        """

        # e-greedy
        if random.random() < epsilon:
            return random.choice(Actions.actions)
        else:
            # recuperate all rules for the current state where the agent in involved
            rules = self.rules.get_rules_with_agent(self.pred_id, state)

            actions = [rule.actions for rule in rules]
            q_values = [rule.scaled_value() for rule in rules]

            # maximum q value
            max_q_value = max(q_values)
            # best actions
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q_value]
            # choose uniformly the best action
            selected = random.choice(best_actions)
            return selected[self.pred_id]

    def q_value(self, state):
        """
        return the Q value of a state
        """
        rules = self.rules.get_rules_with_agent(self.pred_id, state)

        q_values = [rule.scaled_value() for rule in rules]

        # maximum q value
        return max(q_values)
            

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
        rule = self.coord_graph.get_rules_with_agent(self.pred_id,
                                                     state,
                                                     action)[0]

        # retrieve the rule that correspond to the next action-state
        rule_next = self.coord_graph.get_rules_with_agent(self.pred_id,
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
