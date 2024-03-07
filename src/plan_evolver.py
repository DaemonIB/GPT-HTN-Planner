import random
import numpy as np
import time
from openai_api import call_openai_api, log_response
import math
import sys
import Levenshtein

# Constants
EPSILON = sys.float_info.epsilon
PLAN_SIZE = 25
GRID_SIZE = int(math.sqrt(PLAN_SIZE))
# Target z-score threshold, the desired number of std devs above the mean a score needs to be for the program to end
TARGET_Z_SCORE = 2
# The minimum number of generations needed to begin check if the z-score produced a good result
MIN_TARGET_GENERATION = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.5
NEIGHBORHOOD_SIZE = 3
ADAPTATION_THRESHOLD = 0.1
PROGRESS_DELAY = 0.5  # Added to introduce a small delay between generations
lower_threshold = 10  # Adjust based on the problem and the expected range of diversity values
upper_threshold = 20  # Adjust based on the problem and the expected range of diversity values
mutation_rate_adaptation_factor = 0.1  # A factor to control how fast the mutation rate changes
crossover_rate_adaptation_factor = 0.1  # A factor to control how fast the crossover rate changes

min_mutation_rate = 0.01
max_mutation_rate = 0.5
min_crossover_rate = 0.1
max_crossover_rate = 0.9

def generate_initial_plans(user_goal, plan_size):
    print("Generating initial plans...")
    initial_plans = []

    for i in range(plan_size):
        plan = call_openai_api(
            f"Generate a diverse plan related to solving the following problem: '{user_goal}'. "
            f"Consider the problem domain, any constraints or limitations, and the desired format of the solution.",
            max_tokens=500,
            temperature=1.0,
            strip=True
        )
        initial_plans.append(plan)
        print(f"Generated plan {i}")

    return initial_plans

def create_toroidal_grid(population, grid_size):
    second_dimension = PLAN_SIZE // grid_size
    grid = np.array(population).reshape(grid_size, second_dimension)
    return grid

def get_neighborhood(grid, x, y, neighborhood_size):
    neighbors = []
    for i in range(-neighborhood_size // 2, neighborhood_size // 2 + 1):
        for j in range(-neighborhood_size // 2, neighborhood_size // 2 + 1):
            if i == 0 and j == 0:
                continue
            nx, ny = (x + i) % grid.shape[0], (y + j) % grid.shape[1]
            neighbors.append(grid[nx, ny])
    return neighbors

def mutate_plan(plan):
    mutated_plan = call_openai_api(
        f"Modify the following plan to make it more effective: '{plan}'",
        max_tokens=500,
        temperature=1.0,
        strip=True
    )

    return mutated_plan.strip()

def llm_crossover(parent_1, parent_2):
    child_plan = call_openai_api(
        f"Create a new plan by combining the best features of the following two plans: '{parent_1}' and '{parent_2}'",
        max_tokens=500,
        temperature=0.5,
        strip=True
    )

    return child_plan.strip()

def fitness_score(plan, neighbors, memoized_scores, user_goal):
    print("Calculating fitness scores...")

    if plan in memoized_scores:
        score = memoized_scores[plan]
    else:
        score_str = call_openai_api(
            f"Rate the quality of the plan '{plan}' for the problem '{user_goal}'. "
            f"Rate it on a scale from 0 to 1, where 0 represents a poor plan and 1 represents an excellent plan:",
            max_tokens=10,
            temperature=0.5,
            strip=True
        )

        try:
            score = float(score_str)
        except ValueError:
            print(f"Error: Unable to convert '{score_str}' to float for plan: {plan}")
            score = 0

        memoized_scores[plan] = score

    neighbor_scores = [memoized_scores.get(neighbor, 0) for neighbor in neighbors]

    return float(score), sum(neighbor_scores) / len(neighbor_scores)

def calculate_fitness_stats(fitness_scores):
    average_fitness = np.mean(fitness_scores)
    std_dev_fitness = np.std(fitness_scores)
    z_scores = (fitness_scores - average_fitness) / std_dev_fitness
    return z_scores, average_fitness, std_dev_fitness

def roulette_wheel_selection(neighbors, memoized_scores):
    total_fitness = sum([memoized_scores.get(neighbor, 0) for neighbor in neighbors])
    selection_point = random.uniform(0, total_fitness)
    current_sum = 0

    for neighbor in neighbors:
        current_sum += memoized_scores.get(neighbor, 0)
        if current_sum >= selection_point:
            return neighbor

    return neighbors[-1]

# Smoothly transition between a highly connected and locally connected topology based on the std dev
def adapt_neighborhood_size(neighborhood_size, std_dev_fitness):
    if std_dev_fitness > ADAPTATION_THRESHOLD:
        new_neighborhood_size = neighborhood_size + 1
    else:
        new_neighborhood_size = max(neighborhood_size - 1, NEIGHBORHOOD_SIZE)

    return new_neighborhood_size

def levenshtein_distance(s1, s2):
    return Levenshtein.distance(s1, s2)

# Calculates the population diversity using the average pairwise Levenshtein distance
def calculate_population_diversity(population):
    total_distance = 0
    num_pairs = 0

    for i in range(len(population)):
        for j in range(i+1, len(population)):
                total_distance += levenshtein_distance(population[i], population[j])
                num_pairs += 1

    average_distance = total_distance / num_pairs if num_pairs > 0 else 0
    return average_distance

def update_mutation_crossover_rates(mutation_rate, crossover_rate, diversity, min_mutation_rate, max_mutation_rate, min_crossover_rate, max_crossover_rate):
    if diversity < lower_threshold:
        mutation_rate = min(max_mutation_rate, mutation_rate * (1 + mutation_rate_adaptation_factor))
        crossover_rate = max(min_crossover_rate, crossover_rate * (1 - crossover_rate_adaptation_factor))
    elif diversity > upper_threshold:
        mutation_rate = max(min_mutation_rate, mutation_rate * (1 - mutation_rate_adaptation_factor))
        crossover_rate = min(max_crossover_rate, crossover_rate * (1 + crossover_rate_adaptation_factor))

    return mutation_rate, crossover_rate

def main(user_goal):
    initial_plans = generate_initial_plans(user_goal, PLAN_SIZE)
    grid = create_toroidal_grid(initial_plans, GRID_SIZE)
    generation = 0
    memoized_scores = {}

    neighborhood_size = NEIGHBORHOOD_SIZE
    mutation_rate = MUTATION_RATE
    crossover_rate = CROSSOVER_RATE

    fitness_scores = np.full(grid.shape, EPSILON) + np.random.uniform(-0.01, 0.01, grid.shape)

    while True:
        generation += 1
        print(f"Generation: {generation}\n")

        # Calculate the diversity of the population after each generation and update the mutation and crossover rates accordingly.
        population = grid.flatten()  # Convert the 2D grid to a 1D array
        diversity = calculate_population_diversity(population)
        # Updates mutation and crossover rates based on the diversity.
        mutation_rate, crossover_rate = update_mutation_crossover_rates(mutation_rate, crossover_rate, diversity, min_mutation_rate, max_mutation_rate, min_crossover_rate, max_crossover_rate)

        # Calculate fitness statistics
        z_scores, average_fitness, std_dev_fitness = calculate_fitness_stats(fitness_scores)
        print(f"Average fitness: {average_fitness}\nStandard deviation of fitness: {std_dev_fitness}\n")

        # Adapt neighborhood size based on fitness statistics
        neighborhood_size = adapt_neighborhood_size(neighborhood_size, std_dev_fitness)
        print(f"Adapted neighborhood size: {neighborhood_size}\n")

        neighbor_scores = np.zeros(grid.shape)

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                neighbors = get_neighborhood(grid, x, y, neighborhood_size)
                fitness_scores[x, y], neighbor_scores[x, y] = fitness_score(grid[x, y], neighbors, memoized_scores, user_goal)

        max_fitness = np.max(z_scores)
        print(f"Max fitness: {max_fitness}")

        best_plan = grid[np.unravel_index(np.argmax(z_scores), grid.shape)]
        print(f"Current best plan: {best_plan}\nFitness: {max_fitness}\n")

        log_response("best_plan", best_plan)

        # Determine if the current "max_fitness" is "TARGET_Z_SCORE" std deviations above the mean
        if max_fitness >= TARGET_Z_SCORE and generation > MIN_TARGET_GENERATION:
            break

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                neighbors = get_neighborhood(grid, x, y, neighborhood_size)
                parent_1 = grid[x, y]
                parent_2 = roulette_wheel_selection(neighbors, memoized_scores)

                if random.random() < crossover_rate:
                    child = llm_crossover(parent_1, parent_2)
                else:
                    child = mutate_plan(parent_1) if random.random() < 0.5 else mutate_plan(parent_2)

                grid[x, y] = child

        print(f"Progress: Generation {generation} completed. Moving to the next generation...\n")
        time.sleep(PROGRESS_DELAY)  # Added to introduce a small delay between generations

    print(f"Final best plan: {best_plan}\nFitness: {max_fitness}")

if __name__ == "__main__":
    user_goal = input("Enter your goal/problem: ")
    main(user_goal)