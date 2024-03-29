import random
import numpy as np
import time
from openai_api import call_openai_api, log_response
import math
import sys
import Levenshtein
import gensim.downloader as api

# Constants
EPSILON = sys.float_info.epsilon
PROMPT_SIZE = 25
GRID_SIZE = int(math.sqrt(PROMPT_SIZE))
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

# Load pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Normalize the vectors (important for speeding up WMD computation)
word2vec_model.init_sims(replace=True)

def cosine_similarity(text1, text2):
    # Compute the Word Mover's Distance between the two texts
    distance = word2vec_model.wmdistance(text1, text2)
    return 1 / (1 + distance)

def generate_initial_prompts(user_goal, prompt_size):
    print("Generating initial prompts...")
    initial_prompts = []

    for i in range(prompt_size):
        prompt = call_openai_api(
            f"Generate a diverse prompt related to solving the following problem: '{user_goal}'. "
            f"Consider the problem domain, any constraints or limitations, and the desired format of the solution.",
            max_tokens=500,
            temperature=1.0,
            strip=True
        )
        initial_prompts.append(prompt)
        print(f"Generated prompt {i}")

    return initial_prompts

def create_toroidal_grid(population, grid_size):
    second_dimension = PROMPT_SIZE // grid_size
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

def mutate_prompt(prompt):
    mutated_prompt = call_openai_api(
        f"Modify the following prompt to make it more effective: '{prompt}'",
        max_tokens=500,
        temperature=1.0,
        strip=True
    )

    return mutated_prompt.strip()

def llm_crossover(parent_1, parent_2):
    child_prompt = call_openai_api(
        f"Create a new prompt by combining the best features of the following two prompts: '{parent_1}' and '{parent_2}'",
        max_tokens=500,
        temperature=0.5,
        strip=True
    )

    return child_prompt.strip()

def generate_result(prompt):
    result = call_openai_api(
        prompt,
        max_tokens=500,
        temperature=0.5,
        strip=True
    )

    return result

def fitness_score(prompt, neighbors, memoized_scores, user_goal):
    print("Calculating fitness scores...")

    if prompt in memoized_scores:
        score = memoized_scores[prompt]
    else:
        result = generate_result(prompt)
        score_str = call_openai_api(
            f"Rate the quality of the solution '{result}' for the problem '{user_goal}'. "
            f"Rate it on a scale from 0 to 1, where 0 represents a poor solution and 1 represents an excellent solution:",
            max_tokens=10,
            temperature=0.5,
            strip=True
        )
        # score_str = cosine_similarity(result, user_goal)

        try:
            score = float(score_str)
        except ValueError:
            print(f"Error: Unable to convert '{score_str}' to float for prompt: {prompt}")
            score = 0

        memoized_scores[prompt] = score

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
    initial_prompts = generate_initial_prompts(user_goal, PROMPT_SIZE)
    grid = create_toroidal_grid(initial_prompts, GRID_SIZE)
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

        best_prompt = grid[np.unravel_index(np.argmax(z_scores), grid.shape)]
        best_result = generate_result(
            f"{best_prompt} Consider the problem domain, any constraints or limitations, and the desired format of the solution."
            )
        print(f"Current best prompt: {best_prompt}\nFitness: {max_fitness}\nBest result: {best_result}\n")

        log_response("best_prompt", best_prompt)
        log_response("best_result", best_result)

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
                    child = mutate_prompt(parent_1) if random.random() < 0.5 else mutate_prompt(parent_2)

                grid[x, y] = child

        print(f"Progress: Generation {generation} completed. Moving to the next generation...\n")
        time.sleep(PROGRESS_DELAY)  # Added to introduce a small delay between generations

    print(f"Final best prompt: {best_prompt}\nFitness: {max_fitness}\nBest result: {best_result}")

if __name__ == "__main__":
    user_goal = input("Enter your goal/problem: ")
    main(user_goal)