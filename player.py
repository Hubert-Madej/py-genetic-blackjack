import random
from torch import Tensor, argmax, cat, rand, rand_like

import neural_network
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
        input_data = input_data.view(-1, 3)
        output = self.strategy(input_data)
        return argmax(output).item()

    def mutate(self, mutation_rate=0.5):
        for param in self.strategy.parameters():
            if rand(1).item() < mutation_rate:
                param.data += rand_like(param.data) * 0.1
        return self

    def crossover(self, other: 'Player') -> ('Player', 'Player'):
        child1 = Player(neural_network.NeuralNetwork())
        child2 = Player(neural_network.NeuralNetwork())

        for param1, param2, param_self, param_other in zip(child1.strategy.parameters(), child2.strategy.parameters(),
                                                           self.strategy.parameters(), other.strategy.parameters()):
            # Flatten parameters to 1D for easy crossover
            param1_data = param_self.data.view(-1)
            param2_data = param_other.data.view(-1)

            crossover_point = random.randint(0, param1_data.size(0))

            # Perform crossover
            param1.data = cat((param1_data[:crossover_point], param2_data[crossover_point:]), 0).view(
                param_self.data.shape)
            param2.data = cat((param2_data[:crossover_point], param1_data[crossover_point:]), 0).view(
                param_other.data.shape)

        return child1, child2
