import math
import random
import numpy
import pandas
import matplotlib.pyplot as pyplot
from deap import algorithms, base, creator, tools
from time import time
from typing import List, Tuple, Callable, TypeVar
from IPython.display import display

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', None)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

T = TypeVar('T')

def evolution(*,
              individual_generator: Callable[[], T],
              fitness_evaluation: Callable[[T], float],
              population_size: int,
              generations: int,
              crossover_rate: float,
              mutation_rate: float,
              mutation_function: Tuple[Callable, dict],
              ):

    start_time = time()

    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initIterate, creator.Individual, individual_generator)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda individual: (fitness_evaluation(individual),))

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutation_function[0], **mutation_function[1])

    toolbox.register("select", tools.selTournament, tournsize=4)

    stats = tools.Statistics(key=lambda individual: individual.fitness.values)
    stats.register("min", lambda population: numpy.min([fitness for fitness in population if fitness[0] != math.inf]))
    stats.register("avg", lambda population: numpy.mean([fitness for fitness in population if fitness[0] != math.inf]))
    stats.register("max", lambda population: numpy.max([fitness for fitness in population if fitness[0] != math.inf]))

    hall_of_fame = tools.HallOfFame(maxsize=1)

    _, log = algorithms.eaSimple(
        toolbox.population(n=population_size),
        toolbox,
        ngen=generations,
        cxpb=crossover_rate, mutpb=mutation_rate,
        stats=stats, halloffame=hall_of_fame, verbose=True,
    )

    return time() - start_time, log, hall_of_fame[0]

def plot_generations(generation: List[int], average: List[float], minimum: List[int], maximum: List[int]):
    pyplot.figure(figsize=(16, 8))
    pyplot.grid(True)
    # pyplot.plot(generation, average, label="Average")
    pyplot.plot(generation, minimum, label="Mínimo")
    # pyplot.plot(generation, maximum, label="Máximo")
    pyplot.xlabel("Generation")
    pyplot.ylabel("Fitness")
    pyplot.ylim(-1)
    pyplot.legend(loc="upper right")
    pyplot.show()

def display_positional_grid(individual: List[int]):
    dimension = len(individual)

    board = pandas.DataFrame("", index = range(1, dimension + 1), columns = range(1, dimension + 1))

    for x in range(dimension):
        x_row, x_column = individual[x] // dimension, individual[x] % dimension
        for y in range(x + 1, dimension):
            y_row, y_column = individual[y] // dimension, individual[y] % dimension

            diff_row, diff_column = y_row - x_row, y_column - x_column

            if x_row == y_row or x_column == y_column or abs(diff_row) == abs(diff_column):
                for i in range(1 + max(abs(diff_row), abs(diff_column))):
                    board[1 + x_column + i * numpy.sign(diff_column)][1 + x_row + i * numpy.sign(diff_row)] = "🟥"

    for queen in individual:
        row, column = queen // dimension, queen % dimension
        board[1 + column][1 + row] = "👑" if board[1 + column][1 + row] == "" else "♕"

    display(board)

def evaluate_position_indexed_fitness(individual: List[int]) -> float:
    if len(individual) != len(set(individual)):
        return math.inf

    dimension = len(individual)
    fitness: float = 0
    for x in range(len(individual)):
        x_row, x_column = individual[x] // dimension, individual[x] % dimension
        for y in range(x + 1, len(individual)):
            y_row, y_column = individual[y] // dimension, individual[y] % dimension
            if x_row == y_row or x_column == y_column or abs(x_row - y_row) == abs(x_column - y_column):
                fitness += 1
    return fitness

duration, log, fittest_individual_p8 = evolution(
    individual_generator=lambda: random.choices(range(8**2), k=8),
    fitness_evaluation=evaluate_position_indexed_fitness,
    population_size=2500,
    generations=100,
    crossover_rate=.5,
    mutation_rate=.5,
    mutation_function=(tools.mutUniformInt, {"low": 0, "up": 8**2 - 1, "indpb": 1/4}))

print(f"Duração: {duration:.3f} segundos")
plot_generations(*log.select("gen", "avg", "min", "max"))

print(f"Posições: {fittest_individual_p8}")
print(f"Rainhas duplicadas: {len(fittest_individual_p8) - len(set(fittest_individual_p8))}")
print(f"Fitness: {abs(fittest_individual_p8.fitness.values[0])}")
display_positional_grid(fittest_individual_p8)