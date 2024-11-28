import random
from typing import List
from aerialist.px4.drone_test import DroneTest
from aerialist.px4.obstacle import Obstacle
from testcase import TestCase
import numpy as np


class GAGenerator(object):
    min_size = Obstacle.Size(2, 2, 15)
    max_size = Obstacle.Size(20, 20, 25)
    min_position = Obstacle.Position(5, 5, 0, 0)
    max_position = Obstacle.Position(50, 50, 0, 90)
    min_obs = 1
    max_obs = 3

    def __init__(self, case_study_file: str, population_size=6) -> None:
        self.case_study = DroneTest.from_yaml(case_study_file)
        # Hyperparameters
        self.pop_size = population_size # each generation's population size, should be EVEN because of crossover
        self.crossover_rate = 1 # [0,1], Used to control the probability of crossover
        self.indiv_mutate_rate = 1 # [0,1], Used to control the probability of mutate happens on Individual
        self.gene_mutate_rate = 1 # [0,1], Used to control the probability of mutate on obstacle parameter gene
        self.rho = 10 # Used to tune how point(sim) is important
        self.gamma = 0.001 # Used to tune how diversity is important
        self.theta = 0.1 # Used to tune how less obstacle is important
        self.output_num = 20 # Used to control how many test cases are final output
        # Storage
        self.all_test_cases = [] # Storing all test cases generated in GA procedure
        self.all_test_dist = [] # Storing all test_cases's min_dist
        self.all_test_score = [] # Storing all test_cases's fitness score
        self.old_pop = [] # Temporary storage for old population(parents) ([chromosome])
        self.new_pop = [None] * self.pop_size # Temporary storage for new population(childs) ([chromosome])
        self.old_test = [] # Temporary storage for old population(parents) ([TestCase])
        self.new_test = [None] * self.pop_size # Temporary storage for new population(parents) ([TestCase])
        self.old_fitness = [] # Temporary storage for old fitness score(parents')
        self.new_fitness = [None] * self.pop_size # Temporary storage for new fitness score(childs')


    def generate(self, gen_budget: int) -> List[TestCase]:
        print(f'GENERATOR - HyperParameter INFO - pop_size = {self.pop_size}, gen_budget = {gen_budget}, \
            crossover_rate = {self.crossover_rate}, indiv_mutate_rate = {self.indiv_mutate_rate}, gene_mutate_rate = {self.gene_mutate_rate}, \
            rho = {self.rho}, gamma = {self.gamma}, theta = {self.theta}, output_num = {self.output_num}')
        # Initialize the SEED generation of population
        print(f'GENERATOR - Initialize - In Generatior Iteration SEED!')
        self.old_pop = self.initialize_first_case()
        print(f'GENERATOR - Initialize - SEED population are: {self.old_pop}')
        for i, indiv in enumerate(self.old_pop):
            # Generate each individual
            print(f'GENERATOR - Initialize - Generation SEED, Individual {i}: Obs_num = {indiv[0]}, \nObs_param_list = {indiv[1]}')
            indiv_test, indiv_min_dist = self.test_gen_exec(indiv)
            self.old_test.append(indiv_test)
            # Initial old_fitness
            indiv_fitness = self.fitness_function(indiv_test, None, indiv_min_dist, indiv[0], diversity=False, less_obs=True)
            self.old_fitness.append(indiv_fitness)

            # Store info
            self.all_test_cases.append(indiv_test)
            self.all_test_dist.append(indiv_min_dist)
            self.all_test_score.append(indiv_fitness)          
            
            print('GENERATOR - Initialize - SEED Generation Generated!')
        
        # Main GA Itertation
        for i in range(gen_budget):
            print('\n----------------------------------------------------------------------------------------------------------------')
            print(f'GENERATOR - MainLoop - In Generatior Iteration {i}!')
            # Crossover on old population(parents)
            crossed_child_pop = self.crossover(self.old_pop)
            print(f'GENERATOR - MainLoop - Iteration {i}\'s Crossover Done!')
            print(f'GENERATOR - MainLoop - crossed_child_pop is {crossed_child_pop}\n')
            
            # Mutate on Crossed population to have child population
            final_child_pop = self.mutate(crossed_child_pop)
            print(f'GENERATOR - MainLoop - Iteration {i}\'s Mutation Done!')
            print(f'GENERATOR - MainLoop - final_child_pop is {final_child_pop}\n')
            
            # child population -> new population
            self.new_pop = final_child_pop
            
            # execute new population
            for j, indiv in enumerate(self.new_pop):
                print(f'GENERATOR - IndivLoop - Excuting Iteration{i}, Individual{j}: Obs_num = {indiv[0]}, \nObs_param_list = {indiv[1]}')
                # Execute and evaluate new individuals
                indiv_test, indiv_min_dist = self.test_gen_exec(indiv)
                indiv_fitness = self.fitness_function(indiv_test, self.old_test[j], indiv_min_dist, indiv[0], diversity=True, less_obs=True)
                self.new_test[j] = indiv_test
                self.new_fitness[j] = indiv_fitness
                # Store info
                self.all_test_cases.append(indiv_test)
                self.all_test_dist.append(indiv_min_dist)
                self.all_test_score.append(indiv_fitness)          
                
            
            # Roulette wheel selection and put them as old_pop(next iteration's parents population)
            self.old_pop, self.old_fitness, self.old_test = self.roulette_wheel_selection(self.new_pop, self.new_fitness, self.new_test)
            print(f'GENERATOR - MainLoop - Iteration {i}\'s Roulette Wheel Selection Done!')
            print(f'GENERATOR - MainLoop - Selected pop is {self.old_pop}\n')

        
        # final sort all test_cases and return most challenging case
        # first zip
        test_with_score = list(zip(self.all_test_cases, self.all_test_score))
        # test_with_score = list(zip(self.all_test_cases, self.all_test_dist)) # Here we can also use test min_dist to sort
        # Filter out None test cases
        filtered_test_with_score = [item for item in test_with_score if item[0] is not None]
        # sort
        sorted_test_with_score = sorted(filtered_test_with_score, key=lambda x:x[1])
        # unzip
        sorted_test_cases = [x[0] for x in sorted_test_with_score]
        sorted_score = [x[1] for x in sorted_test_with_score]
        # pick top cases
        top_n_test_cases = sorted_test_cases[:self.output_num]

        return top_n_test_cases

    def initialize_first_case(self):
        population = []
        for indiv_idx in range(self.pop_size):
            # random pick obstacles number and generate them randomly
            num_obs = random.randint(self.min_obs, self.max_obs)
            obs_param_list = []
            existing_obs = []
            
            for obs_idx in range(num_obs):
                obs_created = False
                for try_idx in range(100):
                    l=random.uniform(self.min_size.l, self.max_size.l)
                    w=random.uniform(self.min_size.w, self.max_size.w)
                    h=random.uniform(self.min_size.h, self.max_size.h)

                    x=random.uniform(self.min_position.x, self.max_position.x)
                    y=random.uniform(self.min_position.y, self.max_position.y)
                    z=0 # obstacles should always be place on the ground
                    r=random.uniform(self.min_position.r, self.max_position.r)
                    # use each obstacle parameter as one gene
                    gene = [l, w, h, x, y, z, r]
                    # print(f'gene = {gene}')
                    # Check Overlap
                    new_obstacle = Obstacle.from_coordinates(gene)
                    # Check if new_obstacle overlaps with existing ones
                    if not any(new_obstacle.intersects(obstacle) for obstacle in existing_obs):
                        existing_obs.append(new_obstacle)
                        obs_param_list.append(gene)
                        obs_created = True
                        break   
                if not obs_created:
                    print("GENERATOR - Initialize - Warning: Could not find a non-overlapping position for an obstacle after 100 tries.")
            # combine num_obs and obs_param_list together as final chromosome
            chromosome = [num_obs, obs_param_list]
            # store chromosome for each individual testcase
            population.append(chromosome)
        # print(f'Inititial population are: {population}')
        return population
    
    def test_gen_exec(self, chromosome):
        # Generate obstacles list
        # read the info from chromosome
        num_obs = chromosome[0]
        obs_param_list = chromosome[1]
        # initialize the list to store Obstacle instances
        obs_list = []
        for gene in obs_param_list:
            size = Obstacle.Size(
                l=gene[0],
                w=gene[1],
                h=gene[2],
                )
            position = Obstacle.Position(
                x=gene[3],
                y=gene[4],
                z=gene[5],
                r=gene[6],
                )
            obstacle = Obstacle(size, position)
            obs_list.append(obstacle) 
        # Generate TestCase
        test = TestCase(self.case_study, obs_list)
        try:
            test.execute()
            distances = test.get_distances()
            min_dist = min(distances)
            print(f"GENERATOR - Execution - Minimum_distance:{min_dist}")
            test.plot()
        except Exception as e:
            test = None
            min_dist = 10000 # really huge dist with point=0.00001
            print("GENERATOR - Execution - Exception during test execution, skipping the test")
            print(e)
        return test, min_dist
    
    def fitness_function(self, test_now=None, test_past=None, min_dist=1000, obs_num_now=0, diversity=False, less_obs=False):
        # Calculate fitness value
        # calculate point(sim) first
        # # Hard Fail
        # if min_dist < 0:
        #     point_sim = 10 
        # # Soft Fail
        # elif min_dist < 0.25:
        #     point_sim = 5
        # elif min_dist < 1:
        #     point_sim = 2
        # elif min_dist < 1.5:
        #     point_sim = 0.00001 # To avoid fitness == 0 then lead to error in roulette wheel selection
        # # Not Fail
        # else:
        #     point_sim = 0.00001
        # Another Approach: Directly Use min dist 
        if test_now == None:
            test_fitness = 0.0001 # Used when error test
        else: 
            distances = test_now.get_distances()
            ave_min_dist = sum(distances) / len(test_now.test.simulation.obstacles) # Average min distance between trajactory to all obstacles
            point_sim = 1 / (min_dist + ave_min_dist + 0.0001)
            print(f"GENERATOR - Fitness - point(sim):{point_sim}")
            test_fitness = self.rho * point_sim
            # calculate fitness score with point and diversity
            if diversity:
                # calculate diversity_score
                if test_past == None : # Used when the first generation and past error test
                    div_score = 0
                else:
                    div_score = test_now.trajectory.dtw_distance(test_past.trajectory) # Use Build-in DTW Distance Measurement
                print(f"GENERATOR - Fitness - div_score:{div_score}")
                # calculate final test fiteness score
                test_fitness += self.gamma * div_score
            if less_obs:
                # calculate less obstacle score
                less_obs_score = 1 / (obs_num_now + 1)
                print(f"GENERATOR - Fitness - less_obs_score:{less_obs_score}")
                # calculate final test fiteness score
                test_fitness += self.theta * less_obs_score
        
        print(f"GENERATOR - Fitness - fitness:{test_fitness}")
        return test_fitness

    def crossover(self, parent_list):
        print('GENERATOR - Crossover - Begin Crossover!')
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
                print(f'GENERATOR - Crossover - No Crossover between {i} and {j}!')
                crossed_child1 = parent1
                crossed_child2 = parent2
            else:
                # crossover
                print(f'GENERATOR - Crossover - Yes Crossover between {i} and {j}!')
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
        print('GENERATOR - Mutator - Begin Mutation!')
        final_child_list = []
        for crossed_child in crossed_child_list:
            # random pointer to decide whether this time mutate this individual
            mutate_pointer = random.random()
            if mutate_pointer > self.indiv_mutate_rate:
                print('GENERATOR - Mutator - No Mutation!')
                # no mutation
                final_child = crossed_child
                final_child_list.append(final_child)
            else:
                # mutate
                print('GENERATOR - Mutator - Yes Mutation!')
                old_num_obs = crossed_child[0]
                old_obs_param_list = crossed_child[1]
                new_num_obs = old_num_obs + random.choice([-1, 1])
                new_num_obs = max(self.min_obs, min(self.max_obs, new_num_obs))  # Keep within range

                # If number of obstacles increased, add random new obstacles
                if new_num_obs > len(old_obs_param_list):
                    print('GENERATOR - Mutator - Generating new obstacles!')
                    new_obs_param_list = old_obs_param_list
                    existing_obs = [Obstacle.from_coordinates(gene) for gene in new_obs_param_list]
                    for new_obs_idx in range(new_num_obs - len(old_obs_param_list)):
                        obs_created = False
                        for try_idx in range(100):
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
                            new_obstacle = Obstacle.from_coordinates(gene)

                            if not any(new_obstacle.intersects(obstacle) for obstacle in existing_obs):
                                # Add new obstacle to the end
                                existing_obs.append(new_obstacle)
                                new_obs_param_list.append(gene)
                                obs_created = True
                                break   
                            else:
                                print(f"GENERATOR - Mutator - Overlap detected when creating new obstacle, recreating.")
                        if not obs_created:
                            print("GENERATOR - Mutator - Warning: Could not find a non-overlapping position for an obstacle after 100 tries.")
                
                # If number of obstacles decreased, random choose from old obstacles
                elif new_num_obs < len(old_obs_param_list):
                    print('GENERATOR - Mutator - Droping obstacles!')
                    new_obs_param_list = random.sample(old_obs_param_list, k=new_num_obs)

                # If the number of obstacles remains the same, mutate each one (At the edge and moving outside of range)
                else:
                    print('GENERATOR - Mutator - Mutating existing obstacles!')
                    # # Here we keep the old obstacles
                    # new_obs_param_list = old_obs_param_list
                    # Here we randomize genes in Obstacles
                    new_obs_param_list = old_obs_param_list
                    existing_obs = [Obstacle.from_coordinates(gene) for gene in new_obs_param_list]
                    for obs_idx in range(new_num_obs):
                        obs_mutated = False
                        for try_idx in range(100):
                            # Mutation on l
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                l_mutated = random.uniform(self.min_size.l, self.max_size.l)
                            # Mutation on w
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                w_mutated = random.uniform(self.min_size.w, self.max_size.w)
                            # Mutation on h
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                h_mutated = random.uniform(self.min_size.h, self.max_size.h)
                            
                            # Mutation on x
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                x_mutated = random.uniform(self.min_position.x, self.max_position.x)
                            # Mutation on y
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                y_mutated = random.uniform(self.min_position.x, self.max_position.y)
                            # No Mutation on z
                            z_mutated = 0
                            # Mutation on r
                            gene_pointer = random.random()
                            if gene_pointer < self.gene_mutate_rate:
                                r_mutated = random.uniform(self.min_position.x, self.max_position.r)    

                            # Create the mutated obstacle
                            mutated_gene = [l_mutated, w_mutated, h_mutated, x_mutated, y_mutated, z_mutated, r_mutated]
                            mutated_obs = Obstacle.from_coordinates(mutated_gene)

                            # Check if the newly mutated obstacle overlaps with other obstacles
                            other_obstacles = existing_obs[:obs_idx] + existing_obs[obs_idx + 1:]
                            if not any(mutated_obs.intersects(obstacle) for obstacle in other_obstacles):
                                # Update the obstacle list if no overlap occurs
                                existing_obs[obs_idx] = mutated_obs
                                new_obs_param_list[obs_idx] = mutated_obs.to_params()
                                obs_mutated = True
                                break  # Exit the while loop as we found a non-overlapping configuration
                            else:
                                print(f"GENERATOR - Mutator - Overlap detected for obstacle {obs_idx}, regenerating parameters.")
                        if not obs_mutated:
                            print("GENERATOR - Mutator - Warning: Could not find a non-overlapping position for an obstacle after 100 tries.")
                final_child = [new_num_obs, new_obs_param_list]
                final_child_list.append(final_child)
        return final_child_list

    def roulette_wheel_selection(self, new_pop, new_fitness, new_test):
        # Make sure fitness list is not empty
        if len(new_fitness) == 0:
            raise ValueError("GENERATOR - RouletteWheelSelector - Fitness values are empty. Ensure fitness has been calculated before selection.")
        
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
        selected_new_test = []
        for _ in range(len(new_pop)):
            # random pointer to select from wheel
            wheel_pointer = random.random()
            for i, cumulative_prob in enumerate(cumulative_probabilities):
                if wheel_pointer <= cumulative_prob:
                    selected_new_population.append(new_pop[i])
                    selected_new_score.append(new_fitness[i])
                    selected_new_test.append(new_test[i])
                    break
        
        return selected_new_population, selected_new_score, selected_new_test
        

