import gymnasium as gym
import torch
from player import Player
from neural_network import NeuralNetwork

env = gym.make("Blackjack-v1")

model = NeuralNetwork()

model.load_state_dict(torch.load("best_individual"))

player = Player(model)
score = 0
for _ in range(300):

    observation, _ = env.reset()
    terminated, truncated = False, False
    while not (terminated or truncated):
        parsed_observation = (float(observation[0]), float(observation[1]), float(observation[2]))
        input_data = torch.tensor(parsed_observation).unsqueeze(0)
        action = player.decide(input_data)
        observation, reward, terminated, truncated, _ = env.step(action)
        if reward == 1:
            score += reward
print(score/300)



