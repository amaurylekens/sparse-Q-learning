import unittest

from coordination_graph import CoordinationGraph

class CoordinationGraphTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(CoordinationGraphTest, self).__init__(*args, **kwargs)

        self.graph = CoordinationGraph({0:2, 1:2, 2:2, 3:2})

        state_0 = 1
        actions_0 = {0:1, 2:0}
        rho_0 = 4
        self.graph.add_rule(state_0, actions_0, rho_0)

        state_1 = 1
        actions_1 = {0:1, 1:0}
        rho_1 = 5
        self.graph.add_rule(state_1, actions_1, rho_1)

        state_2 = 1
        actions_2 = {1:0}
        rho_2 = 2
        self.graph.add_rule(state_2, actions_2, rho_2)

        state_3 = 1
        actions_3 = {2:1, 1:1}
        rho_3 = 5
        self.graph.add_rule(state_3, actions_3, rho_3)

        state_4 = 2
        actions_4 = {2:1, 3:1}
        rho_4 = 10
        self.graph.add_rule(state_4, actions_4, rho_4)

    def test_add_value_rule(self):

        expected = {0: {"state": 1, "actions": {0:1, 2:0},
                        "rho": 4, "id": 0}, 
                    1: {"state": 1, "actions": {0:1, 1:0},
                        "rho": 5, "id": 1},
                    2: {"state": 1, "actions": {1:0},
                        "rho": 2, "id": 2},
                    3: {"state": 1, "actions": {2:1, 1:1},
                        "rho": 5, "id": 3}, 
                    4: {"state": 2, "actions": {2:1, 3:1},
                        "rho": 10, "id": 4}}

        self.assertEqual(self.graph.rules, expected)

    def test_compute_joint_action(self):

        j_action = self.graph.compute_joint_action(1)

        expected_j_action = {0: 1, 1: 0, 2: 0}

        self.assertCountEqual(j_action, expected_j_action)

    def test_get_rules_with_agent(self):

        rules_1 = self.graph.get_rules_with_agent(1,1)
        expected_rules_1 = [{"state": 1, "actions": {0:1, 1:0},
                           "rho": 5, "id": 1},
                          {"state": 1, "actions": {1:0},
                           "rho": 2, "id": 2},
                          {"state": 1, "actions": {2:1, 1:1},
                           "rho": 5, "id": 3}]

        rules_2 = self.graph.get_rules_with_agent(1,2)
        expected_rules_2 = []

        rules_3 = self.graph.get_rules_with_agent(2,2)
        expected_rules_3 = [{"state": 2, "actions": {2:1, 3:1},
                             "rho": 10, "id": 4}]

        rules_4 = self.graph.get_rules_with_agent(1,1, {0:1, 1:0, 2:0, 3:1})
        expected_rules_4 = [{"state": 1, "actions": {0:1, 2:0},
                             "rho": 4, "id": 0},
                            {"state": 1, "actions": {0:1, 1:0},
                             "rho": 5, "id": 1},
                            {"state": 1, "actions": {1:0},
                             "rho": 2, "id": 2}]

        rules_5 = self.graph.get_rules_with_agent(1,1, {0:1, 1:0, 2:0, 3:1})
        expected_rules_5 = [{"state": 1, "actions": {0:1, 1:0},
                             "rho": 5, "id": 1},
                            {"state": 1, "actions": {1:0},
                             "rho": 2, "id": 2}]

        self.assertCountEqual(rules_1, expected_rules_1)
        self.assertCountEqual(rules_2, expected_rules_2)
        self.assertCountEqual(rules_3, expected_rules_3)
        self.assertNotEqual(rules_4, expected_rules_4)
        self.assertEqual(rules_5, expected_rules_5)
