from actions import Actions
from agent import Agent
from game import Game
from rules import Rules, Rule, generate_game_rules


def play_episode(game, predators, epsilon):
    game.reset()
    capture = False
    while not capture:
        state = game.state

        # compute the action of the predators
        j_action = dict()
        for predator in predators:
            j_action[predator.pred_id] = predator.get_action_choice(state, epsilon)

        # play the actions and get the reward and the next state
        next_state, rewards, capture = game.play(j_action)

        q_values = {predator.pred_id: predator.q_value(state) for predator in predators}

        if not capture:
            future_rewards = {predator.pred_id: predator.q_value(next_state) for predator in predators}
        else:
            future_rewards = {predator.pred_id: 0 for predator in predators}

        rules.update_rule_values(state, j_action, rewards, q_values, future_rewards)

    return game.round


if __name__ == '__main__':
    rules = generate_game_rules(10)
    game = Game(nrow=10, ncol=10)

    # learning parameters
    epsilon = 0.2

    predators = [Agent(0, rules), Agent(1, rules)]

    for i in range(1000):
        r = play_episode(game, predators, epsilon)
        print(r)
