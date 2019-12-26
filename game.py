import random
import numpy as np
from operator import add


class Game:
    def __init__(self, initial_states, nrow=10, ncol=10):

        """
        a prey-predators game

        :param initial_states: initial states of the predators
        :param nrow: number of row of the game grid
        :param ncol: number of col of the game grid
        """

        self.states = initial_states
        self.nrow = nrow
        self.ncol = ncol
        self.prey = (0, 0)
        self.event = None

    def play(self, j_action):

        """
        run one episode of the game

        :param j_action: dict with the predators actions
        :return states: current state of the game
        :return reward: reward for each predators
        :return capture: bool, true if the prey is catched
        """

        capture = False

        for pred_id, pred_action in j_action.items():
            # move of the predator
            self.states[pred_id] = self.move(pred_id, pred_action)

        # boolean if there is predators collision
        collision = (self.states[1] == self.states[0])

        # boolean for each predators if there is next to the prey
        next_prey_states = [(1, 0), (9, 0), (0, 1), (0, 9)]
        next_prey = [state in next_prey_states
                     for state in self.states.values()]

        # boolean if one predator catches without the other one nearby
        failed_capture = (self.states[0] == (0, 0) and not next_prey[1] or
                          self.states[1] == (0, 0) and not next_prey[0])

        # boolean for good capture
        good_capture = (self.states[0] == self.prey and next_prey[1] or
                        self.states[1] == self.prey and next_prey[0])

        if collision:
            self.event = "COLLISION"
            reward = {0: -50, 1: -50}
            # randomly position predators
            self.random_position()

        elif failed_capture:
            self.event = "FAILED CAPTURE"
            if self.states[0] == self.prey:
                reward = {0: -5, 1: -0.5}
            if self.states[1] == self.prey:
                reward = {0: -0.5, 1: -5}

            # find the predator on the prey and move it at random
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            c = list(range(len(directions)))
            p = [1/len(directions)]*len(directions)
            direction = directions[np.random.choice(c, p=p)]
            for pred_id, state in self.states.items():
                if state == self.prey:
                    self.states[pred_id] = self.move(pred_id, direction)

        elif good_capture:
            capture = True
            self.event = "GOOD CAPTURE"
            reward = {0: 37.5, 1: 37.5}
            # randomly position predators
            self.random_position()

        else:
            self.event = None
            reward = {0: -0.5, 1: -0.5}

        return (self.states, reward, capture)

    def move(self, pred_id, action):

        """
        move a predator on the grid

        :param pred_id: the id of the predator to move
        :param action: one valid action ((0,1), (1,0), (-1,0), (0,-1))
        :return: the new state of the predator
        """

        state = self.states[pred_id]
        state = tuple(map(lambda x, y: x + y, state, action))
        state = ((state[0] + 10) % 10, (state[1] + 10) % 10)

        return state

    def random_position(self):

        """
        position the predators at random on the grid
        """

        new_states = []
        for state in range(len(self.states)):
            # potentiel new states : all states except prey
            # and others predators position
            pot_new_states = [(i, j) for i in range(10) for j in range(10)]
            pot_new_states.remove(self.prey)
            pot_new_states = [s for s in pot_new_states if s not in new_states]
            # generate new position with uniform distribution
            p = [1/len(pot_new_states) for i in range(len(pot_new_states))]
            index = list(range(len(pot_new_states)))
            new_state = pot_new_states[np.random.choice(index, p=p)]
            new_states.append(new_state)
        self.states = {pred_id: state
                       for pred_id, state in enumerate(new_states)}

    def get_free_neighbor_cells(self):

        """
        return free neighbor cells of the prey
        :return: list of the free neighbor cells
        """

        free_neighbors = [(1,0), (-1,0), (0,1), (0,-1)]

        for state in self.states.values():
            if state in free_neighbors:
                free_neighboors.remove(state)

        return free_neighbors

    def play_prey(self, action):

        # move of the predator relative to the prey
        relative_move = tuple(map(lambda x: (-1)*x, action))
        for pred_id, state in self.states.items():
            self.states[pred_id] = self.move(pred_id, relative_move)

    def print(self):

        """
        print the grid game with the predators and the prey
        """

        predator_0 = (self.states[0][1]*10) + (self.states[0][0])
        predator_1 = ((self.states[1][1])*10) + (self.states[1][0])
        prey = ((self.prey[1])*10) + (self.prey[0])

        if self.event is not None:
            name = self.print_event(self.event)
            m = int(len(name)/2) + 1
            print("{}|{}".format(" "*m, " "*m))
            print("{}|{}".format(" "*m, " "*m))
            print("-"*(len(name)+2))
            print(" {} ".format(name))
            print("-"*(len(name)+2))
            print("{}|{}".format(" "*m, " "*m))
            print("{}|{}".format(" "*m, " "*m))
            print()
            print()

        print("State : {}".format(self.states))

        cell = 0
        line = " " + ("____ "*self.ncol)
        print(line)
        for i in range(self.nrow):
            line = "|" + ("    |"*self.ncol)
            print(line)
            line = "|"
            for j in range(self.ncol):
                if cell == prey:
                    line += "  O |"
                elif (cell == predator_0):
                    line += " P0 |"
                elif (cell == predator_1):
                    line += " P1 |"
                else:
                    line += "    |"
                cell += 1
            print(line)
            line = " " + ("---- "*self.ncol)
            print(line)
        print()
        print()
