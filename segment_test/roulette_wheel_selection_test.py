import pprint
import random

class GAGenerator:
    min_obs = 1
    max_obs = 3
    crossover_rate = 1

    class Obstacle:
        class Size:
            def __init__(self, l, w, h):
                self.l = l
                self.w = w
                self.h = h

        class Position:
            def __init__(self, x, y, z, r):
                self.x = x
                self.y = y
                self.z = z
                self.r = r

    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(5, 5, 0, 0)
    max_position = Obstacle.Position(50, 50, 0, 90)

    def roulette_wheel_selection(self, new_pop, new_fitness):
        # Make sure fitness list is not empty
        if len(new_fitness) == 0:
            raise ValueError("Fitness values are empty. Ensure fitness has been calculated before selection.")
        
        # Calculate total fitness
        total_fitness = sum(new_fitness)

        # Initialize the prabability wheel
        # Singel Probability
        selection_probabilities = [fitness / total_fitness for fitness in new_fitness]
        # Cumulative Probability
        cumulative_probabilities = []
        cumulative_sum = 0.0
        for prob in selection_probabilities:
            cumulative_sum += prob
            cumulative_probabilities.append(cumulative_sum)
        
        # Select pointed individual
        selected_new_population = []
        selected_new_score = []
        for _ in range(len(new_pop)):
            # random pointer to select from wheel
            wheel_pointer = random.random()
            for i, cumulative_prob in enumerate(cumulative_probabilities):
                if wheel_pointer <= cumulative_prob:
                    selected_new_population.append(new_pop[i])
                    selected_new_score.append(new_fitness[i])
                    break
        
        return selected_new_population, selected_new_score
    

# Create GAGenerator instance
ga_generator = GAGenerator()

# mimic new_pop list
new_pop = [
    [2, [[5, 5, 20, 10, 15, 0, 45], [10, 10, 15, 20, 25, 0, 30]]],
    [3, [[6, 6, 18, 15, 20, 0, 60], [8, 8, 25, 30, 35, 0, 90], [7, 7, 17, 40, 45, 0, 10]]],
    [1, [[9, 9, 19, 25, 30, 0, 75]]],
    [2, [[11, 11, 22, 35, 40, 0, 20], [14, 14, 24, 45, 50, 0, 25]]],
    [3, [[12, 12, 23, 50, 55, 0, 40], [13, 13, 21, 55, 60, 0, 35], [10, 10, 19, 30, 40, 0, 25]]],
    [2, [[15, 15, 25, 60, 65, 0, 50], [17, 17, 20, 70, 75, 0, 55]]]
]

# mimic new_fitness
new_fitness = [10, 20, 15, 25, 5, 30]

print("New Population Before Selection:")
pprint.pprint(new_pop)
print("\nNew Fitness Before Selection:")
pprint.pprint(new_fitness)

# Use roulette_wheel_selection
selected_population, selected_fitness = ga_generator.roulette_wheel_selection(new_pop, new_fitness)

print("\nSelected Population After Roulette Wheel Selection:")
pprint.pprint(selected_population)
print("\nSelected Fitness After Roulette Wheel Selection:")
pprint.pprint(selected_fitness)