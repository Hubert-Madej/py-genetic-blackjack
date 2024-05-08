import random

class Player:
  def __init__(self, risk_tolerance=0.5, strategy=None):
      self.__risk_tolerance = risk_tolerance
      self.__strategy = strategy or self.default_strategy()
      self.__fitness = 0
  
  def get_name(self):
    return self.__name
  
  def get_strategy(self):
    return self.__strategy
  
  def get_fitness(self) -> int:
    return self.__fitness
  
  def give_reward(self, reward: int):
    self.__fitness += reward
    
  def get_risk_tolerance(self) -> float:
    return self.__risk_tolerance
    
  def default_strategy(self):
    strategy = {}
    risk_tolerance = self.get_risk_tolerance()
    
    for player_total in range(4, 13):
      for dealer_card in range(2, 12):
        for usable_ace in range(0, 2):
          if player_total >= 17:
            strategy[(player_total, dealer_card, usable_ace)] = 1 # Hit
          elif player_total <= 11:
            strategy[(player_total, dealer_card, usable_ace)] = 0 # Stick
          else:
            # Adjust strategy for intermediate state according to risk tolerance value.
            if random.random() <= risk_tolerance:
               strategy[(player_total, dealer_card, usable_ace)] = 0 # Stick
            else:
              strategy[(player_total, dealer_card, usable_ace)] = 1 # Hit
    
    return strategy
  
  def decide(self, player_total: int, dealer_card: int, usable_ace: int) -> int:
    key = (player_total, dealer_card, usable_ace)
    strategy = self.get_strategy()
    
    return strategy.get(key, 0) # If combination is not present in strategy - default to Stick.
  
  def mutate(self, mutation_rate = 0.5):
    strategy = self.get_strategy()
    for key in strategy:
      if random.random() < mutation_rate and random.random() < self.get_risk_tolerance():
        strategy[key] = 1 - strategy[key]
        
  def crossover(self, other: 'Player') -> 'Player':
    strategy = self.get_strategy()
    other_strategy = other.get_strategy()
    risk_tolerance = self.get_risk_tolerance()
    child_strategy = {}
    for key in strategy:
      # More aggresive crossover
      if random.random() < risk_tolerance:
        child_strategy[key] = random.choice([strategy[key], other_strategy[key], 0])
      else:
        child_strategy[key] = random.choice([strategy[key], other_strategy[key]])
        
    return Player(risk_tolerance, child_strategy)