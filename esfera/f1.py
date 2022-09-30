from deap import base
from deap import creator
from deap import tools
from deap import benchmarks
from deap import algorithms
import random
import matplotlib.pyplot as pyplot
import numpy

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 6)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", benchmarks.sphere)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def plot_generations(generation, average, minimum, maximum):
    pyplot.figure(figsize=(16, 8))
    pyplot.grid(True)
    pyplot.plot(generation, average, label="average")
    pyplot.plot(generation, minimum, label="minimum")
    pyplot.plot(generation, maximum, label="maximum")
    pyplot.xlabel("Generation")
    pyplot.ylabel("Fitness")
    pyplot.ylim(-1)
    pyplot.legend(loc="upper right")
    pyplot.show()

def main():
    random.seed(64)
    
    pop = toolbox.population(n=10)
    CXPB, MUTPB, GEN = 0.5, 0.2, 100

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", numpy.mean)
    stats.register("Std", numpy.std)
    stats.register("Min", numpy.min)
    stats.register("Max", numpy.max)

    hof = tools.HallOfFame(1)

    _, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=GEN, stats=stats,
                                     halloffame=hof, verbose=True)

    record = stats.compile(pop)
    print("\nMínimo da função:", record["Min"])

    plot_generations(*logbook.select("gen", "Avg", "Min", "Max"))

    return pop, stats, hof

if __name__ == "__main__":
    main()

