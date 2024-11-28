import pprint
import random

class GAGenerator:
    min_obs = 1
    max_obs = 3
    indiv_mutate_rate = 1
    gene_mutate_rate = 0.5

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
    
    def mutate(self, crossed_child_list):
        final_child_list = []
        for crossed_child in crossed_child_list:
            # random pointer to decide whether this time mutate this individual
            mutate_pointer = random.random()
            if mutate_pointer > self.indiv_mutate_rate:
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
                    # # Here we keep the old obstacles
                    # new_obs_param_list = old_obs_param_list
                    # Here we randomize genes in Obstacles
                    new_obs_param_list = old_obs_param_list
                    for obs_idx in range(new_num_obs):
                        # Mutation on l
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][0] = random.uniform(self.min_size.l, self.max_size.l)
                        # Mutation on w
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][1] = random.uniform(self.min_size.w, self.max_size.w)
                        # Mutation on h
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][2] = random.uniform(self.min_size.h, self.max_size.h)
                        
                        # Mutation on x
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][3] = random.uniform(self.min_position.x, self.max_position.x)
                        # Mutation on y
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][4] = random.uniform(self.min_position.x, self.max_position.y)
                        # No Mutation on z
                        new_obs_param_list[obs_idx][4] = 0
                        # Mutation on r
                        gene_pointer = random.random()
                        if gene_pointer < self.gene_mutate_rate:
                            new_obs_param_list[obs_idx][6] = random.uniform(self.min_position.x, self.max_position.r)    

                final_child = [new_num_obs, new_obs_param_list]
                final_child_list.append(final_child)
        return final_child_list

# Create GAGenerator instance
ga_generator = GAGenerator()

# mimic crossed_child_list
crossed_child_list = [
    [2, [[5, 5, 20, 10, 15, 0, 45], [10, 10, 15, 20, 25, 0, 30]]],
    [3, [[6, 6, 18, 15, 20, 0, 60], [8, 8, 25, 30, 35, 0, 90], [7, 7, 17, 40, 45, 0, 10]]]
]

print("Before Mutation:")
pprint.pprint(crossed_child_list)

# Use mutate
mutated_child_list = ga_generator.mutate(crossed_child_list)

print("After Mutation:")
pprint.pprint(mutated_child_list)
