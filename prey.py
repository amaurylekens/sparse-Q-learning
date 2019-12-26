import random

class Prey:
    def __init__(self, n_action):

        """
        object which represent the prey in the game

        :param n_action: number of actions of the prey
        """

        self.n_action = n_action

    def get_action_choice(self):

        """
        return the prey action choice (random)
        """
        a_t = random.randint(0, self.n_action-1)

        return a_t
