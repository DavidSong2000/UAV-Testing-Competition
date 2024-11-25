import pprint
import random

class GAGenerator:
    min_obs = 1
    max_obs = 3
    crossover_rate = 1
    mutate_rate = 1

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

    def crossover(self, parent_list):
        # Initialize crossed_child_list
        crossed_child_list = [None] * len(parent_list)
        # keep the new generation same index with the old generation index
        indices = list(range(len(parent_list)))
        random.shuffle(indices)  # shuffle the indices to random pair
        pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        
        # crossover between two parents in pairs
        for i, j in pairs:
            parent1 = parent_list[i]
            parent2 = parent_list[j]
            # random pointer to decide whether this time crossover this pair
            crossover_pointer = random.random()
            if crossover_pointer > self.crossover_rate:
                # no crossover
                crossed_child1 = parent1
                crossed_child2 = parent2
            else:
                # crossover
                child1_num_obs = parent1[0]
                child2_num_obs = parent2[0]
                parents_obs_param_pool = parent1[1] + parent2[1]
                child1_obs_param_list = random.sample(parents_obs_param_pool, k=child1_num_obs)
                child2_obs_param_list = random.sample(parents_obs_param_pool, k=child2_num_obs)
                
                crossed_child1 = [child1_num_obs, child1_obs_param_list]
                crossed_child2 = [child2_num_obs, child2_obs_param_list]
            # put crossed_child to the correct position
            crossed_child_list[i] = crossed_child1
            crossed_child_list[j] = crossed_child2

        return crossed_child_list

    def mutate(self, crossed_child_list):
        final_child_list = []
        for crossed_child in crossed_child_list:
            # random pointer to decide whether this time mutate this individual
            mutate_pointer = random.random()
            if mutate_pointer > self.mutate_rate:
                # no mutation
                final_child = crossed_child
                final_child_list.append(final_child)
            else:
                # mutate
                old_num_obs = crossed_child[0]
                old_obs_param_list = crossed_child[1]
                new_num_obs = old_num_obs + random.choice([-1, 1])
                new_num_obs = max(self.min_obs, min(self.max_obs, new_num_obs))  # Keep within range

                # If number of obstacles increased, add random new obstacles
                if new_num_obs > len(old_obs_param_list):
                    new_obs_param_list = old_obs_param_list
                    for _ in range(new_num_obs - len(old_obs_param_list)):
                        '''
                        Here we regenerate all 7 parameters, 
                        but we can also only regenerate several of them
                        by moving mutate_pointer inside here
                        '''
                        l=random.uniform(self.min_size.l, self.max_size.l)
                        w=random.uniform(self.min_size.w, self.max_size.w)
                        h=random.uniform(self.min_size.h, self.max_size.h)

                        x=random.uniform(self.min_position.x, self.max_position.x)
                        y=random.uniform(self.min_position.y, self.max_position.y)
                        z=0 # obstacles should always be place on the ground
                        r=random.uniform(self.min_position.r, self.max_position.r)
                        # use each obstacle parameter as one gene
                        gene = [l, w, h, x, y, z, r]
                        # Add new obstacle to the end
                        new_obs_param_list.append(gene)
                
                # If number of obstacles decreased, random choose from old obstacles
                elif new_num_obs < len(old_obs_param_list):
                    new_obs_param_list = random.sample(old_obs_param_list, k=new_num_obs)

                # If the number of obstacles remains the same, keep original (At the edge and moving outside of range)
                else:
                    new_obs_param_list = old_obs_param_list    

                final_child = [new_num_obs, new_obs_param_list]
                final_child_list.append(final_child)
        return final_child_list

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

# mimic old_pop list
old_pop = [
    [2, [[5, 5, 20, 10, 15, 0, 45], [10, 10, 15, 20, 25, 0, 30]]],
    [3, [[6, 6, 18, 15, 20, 0, 60], [8, 8, 25, 30, 35, 0, 90], [7, 7, 17, 40, 45, 0, 10]]],
    [1, [[9, 9, 19, 25, 30, 0, 75]]],
    [2, [[11, 11, 22, 35, 40, 0, 20], [14, 14, 24, 45, 50, 0, 25]]],
    [3, [[12, 12, 23, 50, 55, 0, 40], [13, 13, 21, 55, 60, 0, 35], [10, 10, 19, 30, 40, 0, 25]]],
    [2, [[15, 15, 25, 60, 65, 0, 50], [17, 17, 20, 70, 75, 0, 55]]]
]

# mimic old_fitness
old_fitness = [10, 20, 15, 25, 5, 30]

print("Old Population Before GA:")
pprint.pprint(old_pop)
print("New Fitness Before GA:")
pprint.pprint(old_fitness)

# Crossover on old population(parents)
crossed_child_pop = ga_generator.crossover(old_pop)

print('-------------------------------------------------------')
print("Population After Crossover:")
pprint.pprint(crossed_child_pop)

# Mutate on Crossed population to have child population
final_child_pop = ga_generator.mutate(crossed_child_pop)
print('-------------------------------------------------------')
print("Population After Mutation:")
pprint.pprint(final_child_pop)

# child population -> new population
new_pop = final_child_pop

# mimic new_fitness
new_fitness = [20, 15, 5, 10, 5, 25]

# Roulette wheel selection and put them as old_pop(next iteration's parents population)
old_pop, old_fitness = ga_generator.roulette_wheel_selection(new_pop, new_fitness)

print('-------------------------------------------------------')
print("Selected Population After Roulette Wheel Selection:")
pprint.pprint(old_pop)
print("Selected Fitness After Roulette Wheel Selection:")
pprint.pprint(old_fitness)