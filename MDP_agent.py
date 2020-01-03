import random
import numpy as np


class MDPAgent:
    def __init__(self, n_action, ncol, nrow, Q_t):
        self.n_action = n_action
        self.n_joint_action = n_action*n_action
        self.Q_t = Q_t

    @staticmethod
    def get_init_Q_t(n_joint_action, ncol, nrow):
        return {repr(((x1, y1), (x2, y2))) : [0]*n_joint_action for x1 in range(ncol) for y1 in range(nrow) for x2 in range(nrow) for y2 in range(ncol) if not ((x1==x2 and y1==y2) or (x1==0 and y1==0) or (x2==0 and y2==0)) }

    def get_action_choice(self, state, epsilon):

        """
        return the agent's choice of action for a particular state

        :param state: the state of the game
        :param epsilon: e-greedy parameter
        :return: the agent's choice of action
        """

        # e-greedy
        if (random.uniform(0, 1) < epsilon):
            a_t = random.randint(0, self.n_joint_action-1)
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

        return self.index_to_joint_action(a_t)
            

    def make_q_update(self, reward, state, joint_action, next_state, alpha, gamma):

        """
        update the Q-table for the previous state and the chosen joint action

        :param reward: reward received for the last joint action choice
        :param state: the state before the joint action
        :param joint_action: the joint action
        :param next_state: the state resulting of the joint action
        :param alpha: learning rate
        :param gamma: discount factor
        """
        action_index = self.joint_action_to_index(joint_action)
        previous_value = self.Q_t[state][action_index]
        max_future_reward = max(self.Q_t[next_state])
        new_value = reward + gamma*max_future_reward

        self.Q_t[state][action_index] = (1-alpha)*previous_value + alpha*new_value

    def index_to_joint_action(self, index):
        action0 = index // self.n_action
        action1 = index % self.n_action
        return (action0, action1)

    def joint_action_to_index(self, joint_action):
        index = joint_action[0]*self.n_action + joint_action[1]
        return index
    