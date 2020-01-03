import json
import os
from typing import Dict, Tuple

from actions import Actions


class Rule:
    def __init__(self,
                 state: Dict[int, Tuple[int, int]],
                 actions: Dict[int, str],
                 value: float):
        self.state = state
        self.actions = actions
        self.coordination = len(actions)
        self.value = value

    def valid_state(self, state) -> bool:
        if len(self.state) < len(state):
            for i, s in self.state.items():
                if i not in state or s != state[i]:
                    return False
            return True
        else:
            for i, s in state.items():
                if i not in self.state or s != self.state[i]:
                    return False
            return True

    def valid_agent(self, agent) -> bool:
        return agent in self.actions

    def valid_action(self, joint_action) -> bool:
        for i, a in self.actions.items():
            if i not in joint_action or a != joint_action[i]:
                return False
        return True

    def to_dict(self) -> Dict:
        return {'state': self.state,
                'actions': self.actions,
                'value': self.value}

    def scaled_value(self) -> float:
        """
        return: rule value (rho) divided by the number of involved agents
        """
        return self.value / len(self.actions)

    def __repr__(self):
        return self.to_dict().__repr__()


class Rules:
    def __init__(self):
        """
        object that represents the rules
        """

        self._rules: Dict[int, Rule] = dict()

    def add_rule(self, rule: Rule):
        """
        add a rule

        :param rule: rule to add (dict)
        """

        self._rules[len(self._rules)] = rule

    def save_rules(self, name: str = 'rules', directory: str = 'rules_files'):
        """
        saves the rules in json format

        :param directory: directory where to save
        :param name: save file name
        """

        path = os.path.join(directory, f'{name}.json')
        with open(path, 'w', encoding='utf-8') as f:
            content = {i: r.to_dict() for i, r in self._rules.items()}
            json.dump(content, f, ensure_ascii=False, indent=4)

    def load_rules(self, name: str = 'rules', directory: str = 'rules_files'):
        """
        load the rules from a json file

        :param directory: directory where to load
        :param name: load file name
        """
        path = os.path.join(directory, f'{name}.json')
        with open(path) as f:
            data = json.load(f)

        self._rules: Dict[int, Rule] = dict()
        for key, value in data.items():
            rule = Rule(value['state'], value['actions'], value['value'])
            self._rules[int(key)] = rule

    def get_rules_with_agent(self, pred_id, state):
        rules = [rule for rule in self._rules.values()
                 if rule.valid_state(state) and rule.valid_agent(pred_id)]
        coordination = [rule.coordination for rule in rules]
        max_coordination = max(coordination)
        return [rule for rule, coordination in zip(rules, coordination) if coordination == max_coordination]

    def update_rule_values(self, state, j_action, rewards, q_values, future_rewards, alpha=0.3, gamma=0.9):
        rules = [rule for rule in self._rules.values()
                 if rule.valid_state(state) and rule.valid_action(j_action)]
        coordination = [rule.coordination for rule in rules]
        max_coordination = max(coordination)
        rules = [rule for rule, coordination in zip(rules, coordination) if coordination == max_coordination]
        for rule in rules:
            agents = [agent_id for agent_id in j_action if rule.valid_agent(agent_id)]
            rule.value += alpha * sum(rewards[i] + gamma * future_rewards[i] - q_values[i]
                                      for i in agents)


def generate_game_rules(grid_size: int) -> Rules:
    """
    generate rules for the prey-predators game
    """
    rules = Rules()

    # two predators
    predator_ids = [0, 1]

    # generate all possible joint states
    all_joint_states = [{0: (i, j), 1: (k, l)}
                        for i in range(grid_size)
                        for j in range(grid_size)
                        for k in range(grid_size)
                        for l in range(grid_size)]

    # generate all combinations of actions
    all_actions = [{0: i, 1: j} for i in Actions.actions for j in Actions.actions]

    # coordination states
    coordination_states = list()

    # collective rules (the 2 predators are close to the prey)
    for state in all_joint_states:
        if (state[0] != state[1] and
                0 < dist(state[1], (0, 0), grid_size) <= 2 and
                0 < dist(state[0], (0, 0), grid_size) <= 2):
            coordination_states.append(state)
            for action in all_actions:
                rule = Rule(state, action, 75)
                rules.add_rule(rule)

    # collective rules (the two predators are close to each other)
    for state in all_joint_states:
        if (0 < dist(state[0], state[1], grid_size) <= 2 and
                dist(state[1], (0, 0), grid_size) > 0 and
                dist(state[0], (0, 0), grid_size) > 0):

            if state not in coordination_states:
                coordination_states.append(state)
                for action in all_actions:
                    rule = Rule(state, action, 75)
                    rules.add_rule(rule)

    # all individual states
    all_states = [(i, j) for i in range(grid_size) for j in range(grid_size) if (i, j) != (0, 0)]
    # individual rules
    for predator in predator_ids:
        for state in all_states:
            for action in Actions.actions:
                rule = Rule({predator: state}, {predator: action}, 75)
                rules.add_rule(rule)

    return rules


def dist(state_1: Tuple[int, int], state_2: Tuple[int, int], grid_size: int) -> int:
    """
    compute Manhattan distance between states
    """
    dist_0 = min(abs(state_1[0] - state_2[0]), grid_size - abs(state_1[0] - state_2[0]))
    dist_1 = min(abs(state_1[1] - state_2[1]), grid_size - abs(state_1[1] - state_2[1]))
    return dist_1 + dist_0
