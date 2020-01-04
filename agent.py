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
