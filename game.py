import random
from typing import Tuple, List, Union, Dict


class Game:
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)
    STAY = (0, 0)

    def __init__(self, initial_pred_states: Dict[int, Tuple[int, int]], nrow=10, ncol=10):

        """
        a prey-predators game

        :param initial_pred_states: initial states of the predators
        :param nrow: number of row of the game grid
        :param ncol: number of col of the game grid
        """

        self.nrow = nrow
        self.ncol = ncol
        self.predators = initial_pred_states
        self.offset = (0, 0)
        self.event = None

    def play(self, pred_actions):
        """
        run one round of the game

        :param pred_actions: dict with the predators actions
        :return states: current state of the game
        :return reward: reward for each predators
        :return capture: bool, true if the prey is caught
        """
        self.play_prey()
        return self.play_pred(pred_actions)

    def play_pred(self, pred_actions):
        """
        play a round for the predators

        :param pred_actions: dict with the predators actions
        :return states: current state of the game
        :return reward: reward for each predators
        :return capture: bool, true if the prey is caught
        """

        for pred_id, pred_action in pred_actions.items():
            # move each predator
            self.predators[pred_id] = self.add_vectors(self.predators[pred_id], pred_action)

        # if the predators collide
        if self.predators[1] == self.predators[0]:
            self.event = "COLLISION"
            reward = {0: -50, 1: -50}
            # randomly position colliding predators
            self.random_state([0, 1])
            return self.predators, reward, False

        # boolean for each predators if they are next to the prey
        adjacent_states = [(1, 0), ((self.ncol - 1), 0),
                           (0, 1), (0, (self.nrow - 1))]
        next_to_prey = {pred_id: state in adjacent_states
                        for pred_id, state in self.predators.items()}

        # failed capture
        if self.predators[0] == (0, 0) and not next_to_prey[1]:
            self.event = "FAILED CAPTURE"
            self.random_state(0)
            return self.predators, {0: -5, 1: -0.5}, False
        if self.predators[1] == (0, 0) and not next_to_prey[0]:
            self.event = "FAILED CAPTURE"
            self.random_state(1)
            return self.predators, {0: -0.5, 1: -5}, False

        # successful capture
        good_capture = (self.predators[0] == (0, 0) and next_to_prey[1] or
                        self.predators[1] == (0, 0) and next_to_prey[0])
        if good_capture:
            self.event = "GOOD CAPTURE"
            return self.predators, {0: 37.5, 1: 37.5}, True

        # predators moved
        self.event = None
        return self.predators, {0: -0.5, 1: -0.5}, False

    def play_prey(self):
        """
        play a round for the prey
        """
        # remain in the same state
        if random.random() < 0.2:
            return

        # move to an empty cell
        potential_moves = [move for move in [self.UP, self.DOWN, self.LEFT, self.RIGHT]
                           if self.add_vectors((0, 0), move) not in self.predators.values()]
        move = random.choice(potential_moves)
        self.offset = self.add_vectors(self.offset, move)
        minus_move = (-move[0], -move[1])
        for pred_id, state in self.predators.items():
            self.predators[pred_id] = self.add_vectors(state, minus_move)

    def add_vectors(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> Tuple[int, int]:
        """
        add two vectors

        :return sum of the two input vectors
        """
        return (v1[0] + v2[0]) % self.ncol, (v1[1] + v2[1]) % self.nrow

    def random_state(self, pred_ids: Union[int, List[int]]):
        """
        position predators at random on the grid

        :param pred_ids: predator ids of predators to be positioned
        """
        if type(pred_ids) == int:
            pred_ids = [pred_ids]
        potential_states = {(i, j) for i in range(self.nrow) for j in range(self.ncol) if not i == j == 0}
        potential_states -= {state for p, state in self.predators.items() if p not in pred_ids}
        new_states = random.sample(potential_states, k=len(pred_ids))
        for pred_id, state in zip(pred_ids, new_states):
            self.predators[pred_id] = state

    def print(self):
        """
        print the grid game with the predators and the prey
        """

        predator_0 = self.add_vectors(self.predators[0], self.offset)
        predator_0 = predator_0[0] * self.nrow + predator_0[1]
        predator_1 = self.add_vectors(self.predators[1], self.offset)
        predator_1 = predator_1[0] * self.nrow + predator_1[1]
        prey = (self.offset[0] * self.nrow + self.offset[1])

        if self.event is not None:
            print(f"EVENT: {self.event}")

        cell = 0
        line = "  " + "".join([" {}".format(i) for i in range(self.ncol)])
        print(line)
        for i in range(self.nrow):
            line = f"{i} |"
            for j in range(self.ncol):
                if cell == prey:
                    line += "X|"
                elif cell == predator_0:
                    line += "0|"
                elif cell == predator_1:
                    line += "1|"
                else:
                    line += " |"
                cell += 1
            print(line)
        print()
