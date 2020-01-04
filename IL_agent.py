import random
from typing import Dict

from actions import Actions


class ILAgent:
    def __init__(self, pred_id: int, Q_t: Dict[str, Dict[str, float]]):
        self.pred_id = pred_id
        self.Q_t = Q_t

    @staticmethod
    def get_init_Q_t(ncol, nrow):
        return {repr(((x1, y1), (x2, y2))):
                {a: 0 for a in Actions.actions}
                for x1 in range(ncol) for y1 in range(nrow) for x2 in range(nrow) for y2 in range(ncol)
                if not ((x1 == x2 and y1 == y2) or (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0))}

    def get_action_choice(self, state, epsilon) -> str:

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
            # Get the Q-values for the actions in this state
            Qs_t = self.Q_t[state]

            max_Qs_t = max(Qs_t.values())

            # find index of the max Q-values
            max_index = [a for a, q in Qs_t.items()
                         if q == max_Qs_t]

            # choose one of the max-index with uniform distribution
            return random.choice(max_index)

    def make_q_update(self, reward: float, state: str, action: str, next_state: str, alpha: float, gamma: float):

        """
        update the Q-table for the previous state and the chosen action

        :param reward: reward received for the last action choice
        :param state: the state before the action
        :param action: the action
        :param next_state: the state resulting of the action
        :param alpha: learning rate
        :param gamma: discount factor
        """
        previous_value = self.Q_t[state][action]
        if '(0, 0)' in next_state:
            max_future_reward = 0
        else:
            max_future_reward = max(self.Q_t[next_state].values())
        new_value = reward + gamma * max_future_reward

        self.Q_t[state][action] = (1 - alpha) * previous_value + alpha * new_value
