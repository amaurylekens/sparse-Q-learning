import json
import os
from typing import Dict, Tuple


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
        """
        return rules that match the state and the agent
        """
        rules = [rule for rule in self._rules.values()
                 if rule.valid_state(state) and rule.valid_agent(pred_id)]
        coordination = [rule.coordination for rule in rules]
        max_coordination = max(coordination)
        return [rule for rule, coordination in zip(rules, coordination) if coordination == max_coordination]

    def update_rule_values(self, state, j_action, rewards, q_values, future_rewards, alpha=0.3, gamma=0.9):
        """
        update rule values

        :param state git
        :param j_action
        :param rewards
        :param q_values
        :param future_rewards
        :param alpha
        :param gamma

        """
        rules = [rule for rule in self._rules.values()
                 if rule.valid_state(state) and rule.valid_action(j_action)]
        coordination = [rule.coordination for rule in rules]
        max_coordination = max(coordination)
        rules = [rule for rule, coordination in zip(rules, coordination) if coordination == max_coordination]
        for rule in rules:
            agents = [agent_id for agent_id in j_action if rule.valid_agent(agent_id)]
            rule.value += alpha * sum(rewards[i] + gamma * future_rewards[i] - q_values[i]
                                      for i in agents)
