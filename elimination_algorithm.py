import copy
import itertools
import numpy as np
import random

from operator import itemgetter


class EliminationAlgorithm():
    def __init__(self, n_actions, initial_rules, agent_ids):

        """
        object that allow to make the elimination of variables
        and to calculate the best joint action

        :param  n_actions: dict with the number of actions for agents involved
        :param inital_rules: the rules a the coordination graphe
        :param agents_ids: the ids of the agents involved in the elimination
        """

        self.n_actions = n_actions
        self.initial_rules = initial_rules
        self.steps_rules = [initial_rules]
        self.agent_ids = agent_ids

    def forward(self):

        """
        make the forward elimination and save the rules for each step
        """

        # one by one variable elimination
        for id in self.agent_ids[:-1]:
            # recuperate the rules of the last step
            current_rules = self.steps_rules[-1]

            # retrieve all the rules where the agent is involved
            sel_rules = [current_rule for current_rule in current_rules
                         if id in current_rule["actions"].keys()]
            n_sel_rules = [current_rule for current_rule in current_rules
                           if id not in current_rule["actions"].keys()]

            # retrieve the ids of the agents involved in the seleted rules
            sel_agent_ids = []
            for sel_rule in sel_rules:
                for agent_id in sel_rule["actions"].keys():
                    if agent_id not in sel_agent_ids:
                        sel_agent_ids.append(agent_id)

            # If there is rules for only one agent no need of elimination
            if len(sel_agent_ids) < 2:
                self.steps_rules = [sel_rules]
                self.agent_ids = sel_agent_ids
                break

            # n_actions for the selected agent
            n_actions = {sel_agent_id: self.n_actions[sel_agent_id]
                         for sel_agent_id in sel_agent_ids}

            # make elimination
            new_rules = self.elimination(n_actions, sel_rules, id)

            # save the rules for the forward pass
            self.steps_rules.append(n_sel_rules + new_rules)

    def backward(self):

        """
        make the backward pass to compute the best joint action

        :return: the best joint action (dict)
        """

        j_action = dict()

        for step in range(len(self.steps_rules)):
            rules = self.steps_rules[-(step+1)]
            agent_id = self.agent_ids[-(step+1)]

            # compute the payoff for each action of the agent of the step
            payoffs = []
            for action in range(self.n_actions[agent_id]):
                j_action[agent_id] = action
                payoff = 0
                for rule in rules:
                    rule_valided = True
                    for agent, action in rule["actions"].items():
                        if j_action[agent] != action:
                            rule_valided = False
                            break
                    if rule_valided:
                        payoff += rule["rho"]
                payoffs.append(payoff)

            action_choice = payoffs.index(max(payoffs))
            j_action[agent_id] = action_choice

        # add random action for agent without action
        # (when there is no rules for every agent)
        for agent_id in range(2):
            if agent_id not in j_action.keys():
                j_action[agent_id] = random.randint(0, 4)

        return j_action

    def elimination(self, n_actions, rules, agent_id):

        """
        make an elimination step

        :param n_actions: number of actions for the involved agent
        :param rules: rules used for the elimination
        :param agent_id: the id of the agent to eliminate
        """

        n_agents = len(n_actions)

        # compute combination of the action for the agent involved
        # -1 : the agent has no action for that combination
        agents_actions = [(list(range(n_action)) + [-1])
                          for agent, n_action in n_actions.items()
                          if agent != agent_id]
        perms = [list(perm)
                 for perm in list(itertools.product(*agents_actions))]
        perms.remove([-1]*(n_agents-1))

        # for each action of the agent to eliminate compute the payoff for each
        # combination of the actions of the others agents
        n_best_j_actions = []  # best joint action for each action of the agent
        for i in range(n_actions[agent_id]):
            j_actions = []
            for perm in perms:
                j_action = copy.deepcopy(perm)
                j_action.insert(agent_id, i)
                j_action = {i: j_action[i] for i in range(len(j_action))}

                payoff = 0
                # check if the joint action complies with the different rules
                for rule in rules:
                    rule_valided = True
                    for agent, action in rule["actions"].items():
                        if j_action[agent] != action:
                            rule_valided = False
                            break
                    if rule_valided:
                        payoff += rule["rho"]

                j_actions.append((j_action, payoff))

            # find the best joint actions
            max_index = [i for i, j in enumerate(j_actions)
                         if j[1] == max(j_actions, key=itemgetter(1))[1]]
            best_j_actions = itemgetter(*max_index)(j_actions)

            if type(best_j_actions[0]) == tuple:
                # find the less restrictive best joint action (the most -1)
                n = [list(best_j_action[0].values()).count(-1)
                     for best_j_action in best_j_actions]
                best_j_action = best_j_actions[n.index(max(n))]

            else:
                best_j_action = best_j_actions

            n_best_j_actions.append(best_j_action)

        # compute the new rules
        new_rules = []
        n_best_j_action = max(n_best_j_actions, key=itemgetter(1))

        # write the new rule
        best_rule = {"actions": {key: value
                                 for key, value in n_best_j_action[0].items()
                                 if value != -1 and key != agent_id},
                     "rho": n_best_j_action[1]}

        new_rules.append(best_rule)

        # TO DO : clean
        # add the "complementary" rules
        n_worse_j_action = min(n_best_j_actions, key=itemgetter(1))

        agents = list(best_rule["actions"].keys())
        agents_actions = [list(range(n_action)) for agent, n_action
                          in n_actions.items() if agent in agents]
        perms = [list(perm)
                 for perm in list(itertools.product(*agents_actions))]
        perms_to_delete = list(best_rule["actions"].values())
        perms.remove(perms_to_delete)

        for perm in perms:
            for agent, action in zip(agents, perm):
                n_worse_j_action[0][agent] = action
            new_rule = {"actions": {key: value
                                    for key, value
                                    in n_worse_j_action[0].items()
                                    if value != -1 and key != agent_id},
                        "rho": n_worse_j_action[1]}
            new_rules.append(new_rule)

        return new_rules
