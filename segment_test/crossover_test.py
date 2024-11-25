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

    def crossover(self, parent_list):
        # Initialize crossed_child_list
        crossed_child_list = [None] * len(parent_list)
        # keep the new generation same index with the old generation index
        indices = list(range(len(parent_list)))
        random.shuffle(indices)  # shuffle the indices to random pair
        pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        print(f'Pairs are: {pairs}')

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


# Create GAGenerator instance
ga_generator = GAGenerator()

# mimic crossed_child_list
parent_list = [
    [2, [[5, 5, 20, 10, 15, 0, 45], [10, 10, 15, 20, 25, 0, 30]]],
    [3, [[6, 6, 18, 15, 20, 0, 60], [8, 8, 25, 30, 35, 0, 90], [7, 7, 17, 40, 45, 0, 10]]],
    [1, [[9, 9, 19, 25, 30, 0, 75]]],
    [2, [[11, 11, 22, 35, 40, 0, 20], [14, 14, 24, 45, 50, 0, 25]]],
    [3, [[12, 12, 23, 50, 55, 0, 40], [13, 13, 21, 55, 60, 0, 35], [10, 10, 19, 30, 40, 0, 25]]],
    [2, [[15, 15, 25, 60, 65, 0, 50], [17, 17, 20, 70, 75, 0, 55]]]
]

print("Parent List:")
pprint.pprint(parent_list)

# Use crossover
crossed_child_list = ga_generator.crossover(parent_list)

print("New Population After Crossover:")
pprint.pprint(crossed_child_list)