from typing import Tuple, List


class Actions:
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
    STAY = 'stay'
    actions = (UP, DOWN, LEFT, RIGHT, STAY)

    @staticmethod
    def value(action: str) -> Tuple[int, int]:
        if action == Actions.UP:
            return -1, 0
        if action == Actions.DOWN:
            return 1, 0
        if action == Actions.RIGHT:
            return 0, 1
        if action == Actions.LEFT:
            return 0, -1
        if action == Actions.STAY:
            return 0, 0
        raise Exception('Invalid action')

    @staticmethod
    def values(actions: List) -> List[Tuple[int, int]]:
        return [Actions.value(action) for action in actions]
