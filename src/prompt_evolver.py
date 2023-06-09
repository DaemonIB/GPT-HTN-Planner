import random
import numpy as np
import time
from openai_api import call_openai_api, log_response
import math
import sys

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

def main(user_goal):
    initial_prompts = generate_initial_prompts(user_goal, PROMPT_SIZE)
    grid = create_toroidal_grid(initial_prompts, GRID_SIZE)
    generation = 0
    memoized_scores = {}

    neighborhood_size = NEIGHBORHOOD_SIZE
    fitness_scores = np.full(grid.shape, EPSILON) + np.random.uniform(-0.01, 0.01, grid.shape)

    while True:
        generation += 1
        print(f"Generation: {generation}\n")

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

                if random.random() < CROSSOVER_RATE:
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