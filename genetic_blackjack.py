import random
from player import Player
from typing import List

def generate_population(size: int) -> List[Player]:
  population = []
  for _ in range(size):
    risk_tolerance = round(random.uniform(0, 1), 2)
    population.append(Player(risk_tolerance=risk_tolerance))
  
  return population

def select_n_top_players(population: List[Player], n:int):
  sorted_population = sorted(population, key=lambda p: p.get_fitness(), reverse=True)
  
  return sorted_population[:n]

def crossover(player1: Player, player2: Player) -> Player:
  return player1.crossover(player2)

def apply_mutation(population: List[Player], mutation_rate: float):
  for player in population:
    player.mutate(mutation_rate)
    
def evolve_population(population: List[Player], top_n: int, mutation_rate: float):
  top_players = select_n_top_players(population, top_n)
  
  new_population = []
  while len(new_population) < len(population):
    parent1 = random.choice(top_players)
    parent2 = random.choice(top_players)
    
    new_population.append(crossover(player1=parent1, player2=parent2))
  
  apply_mutation(population, mutation_rate)
  
  return new_population