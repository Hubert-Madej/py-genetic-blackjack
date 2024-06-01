import random
from torch import Tensor, argmax, cat, rand, rand_like
from neural_network import NeuralNetwork
class Player:
  def __init__(self, strategy=None):
      self.strategy = strategy
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
    
  def decide(self, input_data: Tensor) -> int:
    output = self.strategy(input_data)

    return argmax(output).item()
    
  def mutate(self, mutation_rate = 0.5):
    for param in self.strategy.parameters():
      if rand(1).item() < mutation_rate:
        param.data += rand_like(param.data) * 0.1
        
  def crossover(self, other: 'Player') -> 'Player':
    child1 = Player(NeuralNetwork())
    child2 = Player(NeuralNetwork())
    
    crossover_point_fc1 = random.randint(2, 14)
    child1.strategy.fc1.bias.data = cat((self.strategy.fc1.weight.data[:crossover_point_fc1], other.strategy.fc1.weight.data[crossover_point_fc1:]), dim=0)
    child2.strategy.fc1.bias.data = cat((self.strategy.fc1.weight.data[:crossover_point_fc1], other.strategy.fc1.weight.data[crossover_point_fc1:]), dim=0)
    
    child1.strategy.fc1.bias.data = cat((self.strategy.fc1.bias.data[:crossover_point_fc1], other.strategy.fc1.bias.data[crossover_point_fc1:]), dim=0)
    child1.strategy.fc1.bias.data = cat((self.strategy.fc1.bias.data[:crossover_point_fc1], other.strategy.fc1.bias.data[crossover_point_fc1:]), dim=0)
    
    crossover_point_fc2 = random.randint(1, 3)
    
    child1.strategy.fc2.bias.data = cat((self.strategy.fc2.weight.data[:crossover_point_fc2], other.strategy.fc2.weight.data[crossover_point_fc2:]), dim=0)
    child2.strategy.fc2.bias.data = cat((self.strategy.fc2.weight.data[:crossover_point_fc2], other.strategy.fc2.weight.data[crossover_point_fc2:]), dim=0)
    
    child1.strategy.fc2.bias.data = cat((self.strategy.fc2.bias.data[:crossover_point_fc2], other.strategy.fc2.bias.data[crossover_point_fc2:]), dim=0)
    child1.strategy.fc2.bias.data = cat((self.strategy.fc2.bias.data[:crossover_point_fc2], other.strategy.fc2.bias.data[crossover_point_fc2:]), dim=0)
    
    return child1, child2