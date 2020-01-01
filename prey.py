import random


class Prey:

    no_move_proba = 0.2

    def __init__(self, n_action):

        """
        object which represent the prey in the game

        :param n_action: number of actions of the prey
        """

        self.n_action = n_action

    def get_action_choice(self, free_cells):

        """
        return the prey action choice (stays on same position with proba 'no_move_proba', else random between free adjacent cells)
        :param free_cells: the neighbor cells
        """
        if (random.uniform(0, 1) < Prey.no_move_proba):
            a_t = (0,0)
        else:
            a_t = free_cells[random.randint(0, len(free_cells)-1)]

        return a_t
