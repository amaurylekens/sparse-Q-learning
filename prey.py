import random


class Prey:
    def __init__(self, n_action):

        """
        object which represent the prey in the game

        :param n_action: number of actions of the prey
        """

        self.n_action = n_action

    def get_action_choice(self, free_cells):

        """
        return the prey action choice (random)
        :param free_cells: the neighbor cells
        """

        a_t = free_cells[random.randint(0, len(free_cells)-1)]

        return a_t
