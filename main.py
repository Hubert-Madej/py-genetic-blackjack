import gymnasium as gym
from player import Player
from genetic_blackjack import generate_population, evolve_population, select_n_top_players
from typing import List

env = gym.make("Blackjack-v1")

GENERATIONS = 200
POPULATION_SIZE = 1000
PLAYER_ROUNDS = 10
TOP_N_TO_KEEP = 3
MUTATION_RATE=0.1

def main():
  population = generate_population(POPULATION_SIZE)
  for generation in range(GENERATIONS):
    simulate_game(population, PLAYER_ROUNDS)
    
    if generation != GENERATIONS - 1:
      population = evolve_population(population, TOP_N_TO_KEEP, MUTATION_RATE)
    
  top_players = select_n_top_players(population, TOP_N_TO_KEEP)
  print("Top Players Fitness:")
  for player in top_players:
    print(f"Win Ratio: {player.get_fitness() / PLAYER_ROUNDS} | Risk Tolerance: {player.get_risk_tolerance()}")
  print()


def simulate_game(population: List[Player], player_rounds: int):
  for player in population:
    observation, _ = env.reset()
    player_total, dealer_card, usable_ace = observation
    for _ in range(player_rounds):
      action = player.decide(player_total, dealer_card, usable_ace)
      observation, reward, terminated, truncated, _ = env.step(action)
      
      player.give_reward(reward)

      if terminated or truncated:
          observation, _ = env.reset()

  env.close()

if __name__ == '__main__':
  main()