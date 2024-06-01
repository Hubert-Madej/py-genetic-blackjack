import random
import statistics
import genetic_blackjack as gb
import pandas as pd

GENERATIONS = 50
POPULATION_SIZE = 300
PLAYER_ROUNDS = 100
TOP_N_TO_KEEP = 3
MUTATION_RATE = 0.3

def main():
    population = gb.generate_population(POPULATION_SIZE)
    mean_fitness_values = []

    for generation in range(GENERATIONS):
        fitnesses = []
        highest_fitness = float('-inf')
        best_individual = None

        for individual in population:
            fitness = gb.compute_fitness(individual, PLAYER_ROUNDS)
            fitnesses.append(fitness)
            if fitness > highest_fitness:
                highest_fitness = fitness
                best_individual = individual

        mean_fitness = round(statistics.mean(fitnesses), 3)
        mean_fitness_values.append(mean_fitness)
        print(f"Generation: {generation + 1}. Mean Fitness: {mean_fitness}")

        total_fitness = sum(fitnesses)
        probabilities = [fitness / total_fitness for fitness in fitnesses]
        next_generation = []

        for _ in range(POPULATION_SIZE // 2):
            parent1 = random.choices(population, weights=probabilities)[0]
            parent2 = random.choices(population, weights=probabilities)[0]
            child1, child2 = gb.crossover(parent1, parent2)
            child1 = gb.mutate(child1, MUTATION_RATE)
            child2 = gb.mutate(child2, MUTATION_RATE)
            next_generation.extend([child1, child2])

        population = next_generation

        population[0] = best_individual


# Save mean fitness values to Excel file
    df = pd.DataFrame({'Generation': list(range(1, GENERATIONS + 1)), 'Mean Fitness': mean_fitness_values})
    df.to_excel('mean_fitness_values.xlsx', index=False)

if __name__ == '__main__':
    main()
