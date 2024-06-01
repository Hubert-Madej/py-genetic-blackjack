import random
import gymnasium as gym
from player import Player
from typing import List
import torch
from neural_network import NeuralNetwork

env = gym.make("Blackjack-v1")


def generate_population(size: int) -> List[Player]:
    population = []
    for _ in range(size):
        strategy = NeuralNetwork()
        population.append(Player(strategy))

    return population


def compute_fitness(player: Player, num_of_games: int):
    score = 0
    for i in range(num_of_games):
        observation, _ = env.reset()
        terminated, truncated = False, False
        while not (terminated or truncated):
            parsed_observation = (float(observation[0]), float(observation[1]), float(observation[2]))
            input_data = torch.tensor(parsed_observation).unsqueeze(0)
            action = player.decide(input_data)
            observation, reward, terminated, truncated, _ = env.step(action)
            if reward == 1:
                score += reward
    return score / num_of_games


def select_n_top_players(population: List[Player], n: int):
    sorted_population = sorted(population, key=lambda p: p.get_fitness(), reverse=True)

    return sorted_population[:n]


def crossover(player1: Player, player2: Player) -> Player:
    return player1.crossover(player2)


def mutate(player: Player, mutation_rate: float):
    return player.mutate(mutation_rate)


def evolve_population(population: List[Player], top_n: int, mutation_rate: float):
    pass
