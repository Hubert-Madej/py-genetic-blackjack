import gymnasium as gym
import random
from player import Player
import genetic_blackjack as gb
from typing import List

GENERATIONS = 5
POPULATION_SIZE = 5
PLAYER_ROUNDS = 1
TOP_N_TO_KEEP = 3
MUTATION_RATE=0.3

def main():
  population = gb.generate_population(POPULATION_SIZE)
  for generation in range(GENERATIONS):
    fitnesses = []
    highest_fitness = float('-inf')
    
    for individual in population:
      fitness = gb.compute_fitness(individual, PLAYER_ROUNDS)
      fitnesses.append(fitness)
      if fitness > highest_fitness:
        highest_fitness = fitness
    
    print(f"Generation: {generation + 1}. Highest Fitness: {highest_fitness}")
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    next_generation = []
    
    for _ in range(POPULATION_SIZE // 2):
      parent1 = random.choices(population, weights=probabilities)[0]
      parent2 = random.choices(population, weights=probabilities)[0]
      child1, child2 = gb.crossover(parent1, parent2)
      child1 = gb.mutate(child1)
      child2 = gb.mutate(child2)
      next_generation.extend([child1, child2])
    
    population = next_generation

if __name__ == '__main__':
  main()