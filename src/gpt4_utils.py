import datetime
import os
from openai_api import call_openai_api, log_response
from text_utils import trace_function_calls
from guidance_prompts import htn_prompts

# Determines if the current world state matches the goal state
@trace_function_calls
def gpt4_is_goal(state, goal_task):
    prompt = (f"Given the current state '{state}' and the goal '{goal_task}', "
              f"determine if the current state satisfies the goal. "
              f"Please provide the answer as 'True' or 'False':")

    response = call_openai_api(prompt)

    log_response("gpt4_is_goal", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip().lower() == "true"

# Provides an initial high level task that is likely to meet the goal requirements to start performing decomposition from
@trace_function_calls
def get_initial_task(goal):
    prompt = f"Given the goal '{goal}', suggest a high level task that will complete it:"

    response = call_openai_api(prompt)
    log_response("get_initial_task", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

@trace_function_calls
def is_task_primitive(task_name, capabilities_text):
    response = htn_prompts.is_task_primitive(task_name, capabilities_text)

    task_type = response.strip()
    return task_type == "primitive"

@trace_function_calls
def compress_capabilities(text):
    prompt = f"Compress the capabilities description '{text}' into a more concise form:"
    response = call_openai_api(prompt)
    return response.choices[0].message.content.strip()

# Needs pre-conditions to prevent discontinuities in the graph
@trace_function_calls
def can_execute(task, capabilities, state):
    prompt = (f"Given the task '{task}', the capabilities '{capabilities}', "
              f"and the state '{state}', determine if the task can be executed. "
              f"Please provide the answer as 'True' or 'False':")

    response = call_openai_api(prompt)

    log_response("can_execute", response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip().lower() == "true"

def log_state_change(prev_state, new_state, task):
    log_dir = "../state_changes"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/state_changes.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Executing task '{task}'\n")
        log_file.write(f"{timestamp}: Previous state: '{prev_state}'\n")
        log_file.write(f"{timestamp}: New state: '{new_state}'\n\n")