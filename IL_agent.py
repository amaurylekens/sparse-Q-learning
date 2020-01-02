import random
import numpy as np


class ILAgent:
    def __init__(self, id, n_action, ncol, nrow):
        self.id = id
        self.n_action = n_action
        self.Q_t = {repr(((x1, y1), (x2, y2))) : [0]*n_action for x1 in range(ncol) for y1 in range(nrow) for x2 in range(nrow) for y2 in range(ncol) if not ((x1==x2 and y1==y2) or (x1==0 and y1==0) or (x2==0 and y2==0)) }
        # print(len(self.Q_t)*5)   # = 48510 with gridsize 10
        
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
            
            # Get the Q-values for the actions in this state
            Qs_t = self.Q_t[state]

            # find index of the max Q-values
            max_index = [i for i, j in enumerate(Qs_t)
                         if j == max(Qs_t)]

            # choose one of the max-index with uniform distribution
            p = [1/len(max_index) for i in range(len(max_index))]
            sel_index = np.random.choice(max_index, p=p)
            a_t = sel_index

        return a_t
            

    def make_q_update(self, reward, state, action, next_state, alpha, gamma):

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
        max_future_reward = max(self.Q_t[next_state])
        new_value = reward + gamma*max_future_reward

        self.Q_t[state][action] = (1-alpha)*previous_value + alpha*new_value

    