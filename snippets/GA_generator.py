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
        self.crossover_rate = 0.5 # [0,1], Used to control the probability of crossover
        self.mutate_rate = 0.1 # [0,1], Used to control the probability of mutate
        self.gamma = 0.8 # Used to tune how diversity is important
        self.theta = 2 # Used to tune how less obstacle is important
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
        # Initialize the first generation of population
        self.old_pop = self.initialize_first_case()
        for indiv in self.old_pop:
            # Generate each individual
            print(f'individual = {indiv}')
            indiv_test, indiv_min_dist = self.test_gen_exec(indiv)
            self.old_test.append(indiv_test)
            
            # Store info
            self.all_test_cases.append(indiv_test)
            self.all_test_dist.append(indiv_min_dist)
            self.all_test_score.append(self.fitness_function(obs_num_now=indiv[0],diversity=False, less_obs=True))          
            
            # Initial old_fitness
            indiv_fitness = self.fitness_function(indiv_test, None, indiv_min_dist, indiv[0], diversity=False, less_obs=True)
            self.old_fitness.append(indiv_fitness)
        
        # Main GA Itertation
        for i in range(gen_budget):
            # Crossover on old population(parents)
            crossed_child_pop = self.crossover(self.old_pop)
            
            # Mutate on Crossed population to have child population
            final_child_pop = self.mutate(crossed_child_pop)
            
            # child population -> new population
            self.new_pop = final_child_pop
            
            # execute new population
            for i, indiv in enumerate(self.new_pop):
                print(f'individual = {indiv}')
                # Execute new individuals
                indiv_test, indiv_min_dist = self.test_gen_exec(indiv)
                self.new_test[i] = indiv_test
                # Store info
                self.all_test_cases.append(indiv_test)
                self.all_test_dist.append(indiv_min_dist)
                self.all_test_score.append(self.fitness_function(indiv_test, self.old_test[i], indiv_min_dist, indiv[0], diversity=True, less_obs=True))          
                # Evaluate new individuals
                self.new_fitness[i] = self.fitness_function(indiv_test, self.old_test[i], indiv_min_dist, , indiv[0], diversity=True, less_obs=True)
            
            # Roulette wheel selection and put them as old_pop(next iteration's parents population)
            self.old_pop, self.old_fitness, self,old_test = self.roulette_wheel_selection(self.new_pop, self.new_fitness, self.new_test)

        
        # final sort all test_cases and return most challenging case
        # first zip
        test_with_score = list(zip(self.all_test_cases, self.all_test_score))
        # test_with_score = list(zip(self.all_test_cases, self.all_test_dist)) # Here we can also use test min_dist to sort
        # sort
        sorted_test_with_score = sorted(test_with_score, key=lambda x:x[1])
        # unzip
        sorted_test_cases = [x[0] for x in sorted_test_with_score]
        sorted_score = [x[1] for x in sorted_test_with_score]
        # pick top cases
        top_n_test_cases = sorted_test_cases[:self.output_num]

        return top_n_test_cases

    def initialize_first_case(self):
        population = []
        for _ in range(self.pop_size):
            # random pick obstacles number and generate them randomly
            num_obs = random.randint(self.min_obs, self.max_obs)
            obs_param_list = []
            for _ in range(num_obs):
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
                obs_param_list.append(gene)      
            # combine num_obs and obs_param_list together as final chromosome
            chromosome = [num_obs, obs_param_list]
            # store chromosome for each individual testcase
            population.append(chromosome)
        print(f'Inititial population are: {population}')
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
            print(f"minimum_distance:{min_dist}")
            # test.plot()
        except Exception as e:
            test = None
            min_dist = 10000 # really huge dist with point=0.00001
            print("Exception during test execution, skipping the test")
            print(e)
        return test, min_dist
    
    def fitness_function(self, test_now=None, test_past=None, min_dist=1000, obs_num_now=0, diversity=False, less_obs=False):
        # Calculate fitness value
        # calculate point(sim) first
        # Hard Fail
        if min_dist < 0:
            point_sim = 10 
        # Soft Fail
        elif min_dist < 0.25:
            point_sim = 5
        elif min_dist < 1:
            point_sim = 2
        elif min_dist < 1.5:
            point_sim = 0.00001 # To avoid fitness == 0 then lead to error in roulette wheel selection
        # Not Fail
        else:
            point_sim = 0.00001
        print(f"point(sim):{point_sim}")
        test_fitness = point_sim
        # calculate fitness score with point and diversity
        if diversity:
            # calculate diversity_score
            if test_past == None or test_now == None: # Used when the first generation and error test
                div_score = 0
            else:
                div_score = test_now.trajectory.dtw_distance(test_past.trajectory) # Use Build-in DTW Distance Measurement
            print(f"div_score:{div_score}")
            # calculate final test fiteness score
            test_fitness += self.gamma * div_score
        if less_obs:
            # calculate less obstacle score
            less_obs_score = 1 / (obs_num_now + 1)
            print(f"less_obs_score:{less_obs_score}")
            # calculate final test fiteness score
            test_fitness += self.theta * less_obs_score
        
        print(f"fitness:{test_fitness}")
        return test_fitness

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

    def roulette_wheel_selection(self, new_pop, new_fitness, new_test):
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
        

