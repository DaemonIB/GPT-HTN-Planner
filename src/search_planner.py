"""
This SearchPlanner class uses a random search approach to create a graph with nodes representing states and edges representing tasks. 
The plan method generates tasks randomly and adds them to the graph if they are executable. 
The weight of the edge between two states is calculated based on the difference between the states and the complexity of the task. 
The planner stops when the goal is reached or the maximum number of iterations is reached.
"""
import heapq
import random
from gpt4_utils import gpt4_is_goal, can_execute, log_state_change
from openai_api import call_openai_api, log_response
from src.guidance_prompts import htn_prompts
from src.guidance_prompts.htn_prompts import calculate_weight
from task_node import TaskNode
from text_utils import extract_lists, trace_function_calls
from graph_manager import GraphManager
import numpy as np

# Constants for weight/cost range
WEIGHT_MIN_VALUE = 0.0
WEIGHT_MAX_VALUE = 100.0

NUM_CRITERIA = 8
MAX_SUGGESTED_VALUE = WEIGHT_MAX_VALUE / NUM_CRITERIA

def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def reconstruct_path(came_from, start, goal):
    path = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# It would be costly and time-consuming, but we might get better results if we estimate each criteria in
# independent calls to the API.
def generate_criteria_prompt():
    criteria_prompt = (
        f"1. Direct performance of the task using capabilities (0-{MAX_SUGGESTED_VALUE}: lower values for better performance, higher values for worse performance). "
        f"2. Action cost in terms of time, resources, energy consumption, or other relevant metrics (0-{MAX_SUGGESTED_VALUE}: lower values for less cost, higher values for more cost). "
        f"3. Risk or uncertainty involved in the task (0-{MAX_SUGGESTED_VALUE}: lower values for less risk, higher values for more risk). "
        f"4. User involvement (0-{MAX_SUGGESTED_VALUE}: lower values for automated solutions, higher values for solutions requiring direct user involvement). "
        f"5. Domain-specific constraints, such as real world actions having higher weights (0-{MAX_SUGGESTED_VALUE}: lower values for fewer constraints, higher values for more constraints). "
        f"6. Historical data or experience, if available (0-{MAX_SUGGESTED_VALUE}: lower values for better past performance, higher values for worse past performance). "
        f"7. Solution quality or preference (0-{MAX_SUGGESTED_VALUE}: lower values for better quality or preference, higher values for worse quality or preference). "
        f"8. Task effectiveness (0-{MAX_SUGGESTED_VALUE}: lower values for tasks that make meaningful progress towards the goal, higher values for tasks that are redundant, illogical, or do not change anything)."
    )
    return criteria_prompt


class SearchPlanner:

    def __init__(self, initial_state, goal_task, capabilities_input, max_iterations, send_update_callback=None):
        self.initial_state = initial_state
        self.goal_task = goal_task
        self.capabilities_input = capabilities_input
        self.max_iterations = max_iterations
        self.graph_manager = GraphManager()
        self.send_update_callback = send_update_callback

        # Add the initial state and goal task to the graph
        self.graph_manager.add_node(initial_state)
        self.graph_manager.add_node(goal_task)

        # Connect the initial state to the goal task with a default weight
        default_weight = float('inf')
        self.graph_manager.add_edge(initial_state, goal_task, default_weight)

    def select_valid_random_tasks(self):
        task_a, task_b = self.graph_manager.select_random_tasks(self.initial_state, self.goal_task)

        return task_a, task_b
    
    def convert_search_plan_to_task_node_plan(self, search_plan):
        if not search_plan:
            return None

        root_node = TaskNode(search_plan[0])
        current_task_node = root_node

        for task in search_plan[1:]:
            task_node = TaskNode(task)
            current_task_node.add_child(task_node)
            current_task_node = task_node

        return root_node

    def print_plan(self, task_node_plan):
            if task_node_plan is None:
                print("No plan found.")
                return

            print("Plan:")
            current_task_node = task_node_plan
            while current_task_node is not None:
                print(current_task_node.task_name)
                current_task_node = current_task_node.children[0] if current_task_node.children else None

    def plan(self):
        # Phase 1: Construct the graph
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1} of {self.max_iterations}")

            task_a, task_b = self.select_valid_random_tasks()

            intermediate_task = self.generate_task(task_a, task_b)
            print(f"Generated task: {intermediate_task}")
            translated_task = self.translate_task(intermediate_task, self.capabilities_input)
            print(f"Translated task: {translated_task}")

            # Provide task_a to the translated task as state to help determine its executability
            if can_execute(translated_task, self.capabilities_input, task_a):
                weight_a_to_intermediate = self.calculate_weight(task_a, intermediate_task, translated_task)
                weight_intermediate_to_b = self.calculate_weight(intermediate_task, task_b, translated_task)
                print(f"Weight from '{task_a}' to '{intermediate_task}': {weight_a_to_intermediate}")
                print(f"Weight from '{intermediate_task}' to '{task_b}': {weight_intermediate_to_b}")

                # Add the intermediate task to the graph and connect it to the selected tasks
                self.graph_manager.add_node(intermediate_task)
                self.graph_manager.add_edge(task_a, intermediate_task, weight_a_to_intermediate)
                self.graph_manager.add_edge(intermediate_task, task_b, weight_intermediate_to_b)

                """
                Remove the edge between task_a and task_b, if one exists
                LLMs do not seem to be good at providing cost estimates that are comparable between 
                larger and smaller tasks. Using intermediary steps seems to produce better results on average.
                """
                if self.graph_manager.has_edge(task_a, task_b) is True:
                    self.graph_manager.delete_edge(task_a, task_b)

        # Phase 2: Perform A* search on the constructed graph
        path = self.astar_search(self.initial_state, self.goal_task)
        # Convert the path into task_nodes so that it can be visualized
        task_node_plan = self.convert_search_plan_to_task_node_plan(path)

        # Print the plan
        self.print_plan(task_node_plan)

        return task_node_plan

    @trace_function_calls
    def generate_task(self, state_a, state_b):
        task = htn_prompts.generate_task(state_a, state_b, self.capabilities_input)
        log_response("generate_task", task)
        return task

    @trace_function_calls
    def translate_task(self, task, capabilities_input):
        translated_task = htn_prompts.translate_task(task, capabilities_input)
        log_response("translate_task", translated_task)
        return translated_task

    @trace_function_calls
    def calculate_weight(self, state_a, state_b, task):
        max_retries = 5
        for attempt in range(max_retries):
            response_str = calculate_weight(state_a, state_b, task, WEIGHT_MIN_VALUE, WEIGHT_MAX_VALUE, generate_criteria_prompt())

            # Check if the response is a valid float
            if is_float(response_str):
                weight = float(response_str)
                log_response("calculate_weight", weight)
                return weight
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError("Failed to convert response to float after multiple attempts.")

    def astar_search(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal:
                return reconstruct_path(came_from, start, goal)

            for next_node, edge_cost in self.graph_manager.get_neighbors(current):
                new_cost = cost_so_far[current] + edge_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.estimate_cost(current, next_node, goal)
                    heapq.heappush(open_list, (priority, next_node))
                    came_from[next_node] = current

        return None

    @trace_function_calls
    def heuristic(self, current, next_node, goal):
        max_retries = 5
        for attempt in range(max_retries):
            response_str = response_str = calculate_weight(next_node, goal, WEIGHT_MIN_VALUE, WEIGHT_MAX_VALUE, generate_criteria_prompt())

            # Check if the response is a valid float
            if is_float(response_str):
                heuristic_cost = float(response_str)
                log_response("heuristic", heuristic_cost)
                return heuristic_cost
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    raise ValueError("Failed to convert response to float after multiple attempts.")

    def estimate_cost(self, current, next_node, goal):
        # Use the weight of the edge between the current and next_node as part of the cost estimation
        edge_weight = self.graph_manager.get_edge_weight(current, next_node)

        # Call the heuristic function to get an admissible heuristic cost
        heuristic_cost = self.heuristic(current, next_node, goal)

        return edge_weight + heuristic_cost
