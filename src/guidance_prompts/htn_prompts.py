import guidance
import os
import time
import outlines.models as models

guidance_gpt4_api = guidance.llms.OpenAI("gpt-4", api_key=os.environ.get('OPENAI_KEY'))
guidance.llm = guidance_gpt4_api
outlines_model = models.text_completion.OpenAICompletion(model_name="gpt-4", max_tokens=3)


def call_guidance_with_retry(guidance_function, max_retries=3, *args, **kwargs):
    retries = 0
    while retries <= max_retries:
        try:
            result = guidance_function(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error: {e}")
            if retries == max_retries:
                raise
            retries += 1
            time.sleep(2 ** retries)  # Exponential backoff

def generate_task(state_a, state_b, capabilities_input):
    good_task_description = (
        "A good task should be relevant, achievable with given capabilities, efficient, low-risk, "
        "minimally dependent on user involvement, and contribute significantly towards the goal."
    )

    task_generation = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the current state '{{state_a}}' and the desired state '{{state_b}}',
    generate a single task that transitions between these states using the following capabilities:
    '{{capabilities_input}}'. Please provide the task in a single line without any commentary
    or superfluous text. Keep in mind the characteristics of a good task: {{good_task_description}}
    {{/user}}

    {{#assistant}}{{gen "task"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    response = call_guidance_with_retry(task_generation, 3, state_a=state_a, state_b=state_b, capabilities_input=capabilities_input, good_task_description=good_task_description)
    task = response['task'].strip()
    return task

# Add new functions for extracting and suggesting new queries here
def extract_and_format_information(webpage_content):
    structured_info = guidance('''
    {{#system~}}You are a helpful assistant.{{~/system}}
    {{#user~}}Extract and format relevant information from the following webpage content: {{webpage_content}}{{~/user}}
    {{#assistant}}{{gen "extracted_info"}}{{/assistant}}''',
                                               llm=guidance_gpt4_api)

    output = call_guidance_with_retry(structured_info, 3, webpage_content=webpage_content)

    return output['extracted_info']


def check_subtasks(task, subtasks, capabilities_input):
    task_statuses = ['True', 'False']

    check_subtasks_program = guidance('''
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}
    {{#user~}}
    Given the parent task '{{task}}', and its subtasks '{{#each subtasks}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}',
    check if these subtasks effectively and comprehensively address the requirements
    of the parent task without any gaps or redundancies, using the following capabilities:
    '{{capabilities_input}}'. Return 'True' if they meet the requirements or 'False' otherwise.
    {{~/user}}
    {{#assistant~}}
    {{select "result" options=task_statuses}}
    {{~/assistant}}''', llm=guidance_gpt4_api)

    response = call_guidance_with_retry(check_subtasks_program, 3, task=task, subtasks=subtasks, capabilities_input=capabilities_input,
                                        task_statuses=task_statuses)
    result = response["result"].strip().lower()

    return result


def get_subtasks(task, state, remaining_decompositions, capabilities_input):
    subtasks_prompt = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the task '{{task}}', the current state '{{state}}',
    and {{remaining_decompositions}} decompositions remaining before failing,
    please decompose the task into a detailed step-by-step plan
    using the following capabilities: '{{capabilities_input}}'. 
    Replace any URLs in your response with the placeholder '[URL]'.
    Provide the subtasks in a comma-separated list,
    each enclosed in square brackets: [subtask1], [subtask2], ...
    {{/user}}
    {{#assistant}}{{gen "subtasks_list"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    result = call_guidance_with_retry(subtasks_prompt, 3, task=task, state=state,
                                      remaining_decompositions=remaining_decompositions,
                                      capabilities_input=capabilities_input)
    subtasks_with_types = result['subtasks_list'].strip()

    return subtasks_with_types


def suggest_new_query(query):
    new_query = guidance('''
    {{#system~}}You are a helpful assistant.{{~/system}}
    {{#user~}}Suggest a new query to find the missing information based on the initial query: {{query}}{{~/user}}
    {{#assistant}}{{gen "new_query"}}{{/assistant}}''',
                                         llm=guidance_gpt4_api)

    output = call_guidance_with_retry(new_query, 3, query=query)

    return output['new_query']


def update_plan_output(task_name, task_description, elapsed_time, time_limit, context_window):
    task_statuses = ['not started', 'in progress', 'completed']
    action_types = ['update', 'insert', 'delete']

    structured_prompt = guidance('''
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}

    {{#user~}}
    Given the task: {{task_name}}, {{task_description}} and the current state of the plan.
    What should be the task status and how should the deliverable be updated?
    Provide instructions to update, insert, or delete lines in the deliverable using quoted text as an indicator.
    The status should be one of the following: 'not started', 'in progress', 'completed'
    Elapsed time: {{elapsed_time}} seconds, time limit: {{time_limit}} seconds, context window: {{context_window}} tokens.
    {{~/user}}
    
    {{#user~}}Status:{{~/user}}
    {{#assistant~}}
    {{select "status" options=task_statuses}}
    {{~/assistant}}
    
    {{#user~}}Action:{{~/user}}
    {{#assistant~}}
    {{select "action" options=action_types}}
    {{~/assistant}}
    
    {{#user~}}Details:{{~/user}}
    {{#assistant~}}
    {{#if (eq action "update")}}{{gen "update_line"}}Update line {{update_line}} with "{{update_text}}"{{/if}}
    {{#if (eq action "insert")}}{{gen "insert_line"}}Insert "{{insert_text}}" at line {{insert_line}}{{/if}}
    {{#if (eq action "delete")}}{{gen "delete_line"}}Delete line {{delete_line}}{{/if}}
    {{~/assistant}}
    ''')

    output = call_guidance_with_retry(structured_prompt, 3,
                                      task_name=task_name,
                                      task_description=task_description,
                                      elapsed_time=elapsed_time,
                                      time_limit=time_limit,
                                      context_window=context_window,
                                      task_statuses=task_statuses,
                                      action_types=action_types
                                      )

    status = output['status']
    action = output['action']
    details = {}

    if action == "update":
        details["update_line"] = output["update_line"]
        details["update_text"] = output["update_text"]
    elif action == "insert":
        details["insert_line"] = output["insert_line"]
        details["insert_text"] = output["insert_text"]
    elif action == "delete":
        details["delete_line"] = output["delete_line"]

    return {"status": status, "action": action, "details": details}


def confirm_deliverable_changes(deliverable_content, updated_content):
    confirm_choices = ['yes', 'no']

    confirm_changes = guidance('''
    {{#system}}You are a helpful agent{{/system}}
    {{#user}}
    Please confirm the changes made to the deliverable.
    Original content:
    {{deliverable_content}}

    Updated content:
    {{updated_content}}

    Type 'yes' to confirm the changes or 'no' to revert them.
    {{/user}}
    {{#assistant}}{{select "confirm" options=confirm_choices}}{{/assistant}}
    ''')

    result = call_guidance_with_retry(confirm_changes, 3, deliverable_content=deliverable_content,
                                      updated_content=updated_content,
                                      confirm_choices=confirm_choices)
    return result['confirm']


def translate(original_task, capabilities_input):
    # translates a task into a form that can be completed with the specified capabilities
    task_translation = guidance('''
    {{#system}}You are a helpful agent{{/system}}
    
    {{#user}}Translate the task '{{task}}' into a form that can be executed using the following capabilities:
    '{{capabilities_input}}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    
    When translated to use the specified capabilities the result is:{{/user}}
    {{#assistant}}{{gen "translated_task"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    result = call_guidance_with_retry(task_translation, 3, task=original_task, capabilities_input=capabilities_input)
    return result['translated_task']

def heuristic(next_node, goal, WEIGHT_MIN_VALUE, WEIGHT_MAX_VALUE, criteria_prompt):
    heuristic_calculation = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Please estimate the remaining cost to reach the goal state '{{goal}}' from the current state '{{next_node}}',
    considering the following criteria:
    {{criteria_prompt}}
    Lower values are considered better. Ensure the response is a float value within the range [{{WEIGHT_MIN_VALUE}}, {{WEIGHT_MAX_VALUE}}].
    {{/user}}

    {{#assistant}}{{gen "heuristic_cost" max_tokens=10}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    response = call_guidance_with_retry(heuristic_calculation, 3, goal=goal, next_node=next_node,
                                        criteria_prompt=criteria_prompt, WEIGHT_MIN_VALUE=WEIGHT_MIN_VALUE,
                                        WEIGHT_MAX_VALUE=WEIGHT_MAX_VALUE)

    return response['heuristic_cost'].strip()

def calculate_weight(state_a, state_b, task, WEIGHT_MIN_VALUE, WEIGHT_MAX_VALUE, criteria_prompt):

    weight_calculation = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Please provide a float value representing the weight of the edge between state '{{state_a}}' and state '{{state_b}}' for the task '{{task}}', considering the following criteria:
    {{criteria_prompt}}
    Lower values are considered better. Ensure the response is a float value within the range [{{WEIGHT_MIN_VALUE}}, {{WEIGHT_MAX_VALUE}}].
    {{/user}}

    {{#assistant}}{{gen "weight" max_tokens=10}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    response = call_guidance_with_retry(weight_calculation, 3, state_a=state_a, state_b=state_b, task=task,
                                        criteria_prompt=criteria_prompt, WEIGHT_MIN_VALUE=WEIGHT_MIN_VALUE,
                                        WEIGHT_MAX_VALUE=WEIGHT_MAX_VALUE)

    response_str = response['weight'].strip()

    return response_str

def translate_task(task, capabilities_input):
    task_translation = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Translate the task '{{task}}' into a form that can be executed using the following capabilities:
    '{{capabilities_input}}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    {{/user}}

    {{#assistant}}{{gen "translated_task"}}{{/assistant}}
    ''', llm=guidance_gpt4_api)

    response = call_guidance_with_retry(task_translation, 3, task=task, capabilities_input=capabilities_input)
    translated_task = response['translated_task'].strip()
    return translated_task

def is_task_primitive(task_name, capabilities_text):
    task_types = ['primitive', 'compound']

    primitive_check = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the task '{{task_name}}' and the capabilities '{{capabilities_text}}',
    determine if the task is primitive which cannot be broken up further or compound which can be broken down more.
    Please provide the answer as 'primitive' or 'compound':
    {{/user}}
    {{~#assistant~}}
    {{select "choice" options=task_types}}
    {{~/assistant~}}
    ''', llm=guidance_gpt4_api)

    result = call_guidance_with_retry(primitive_check, 3, task_name=task_name, capabilities_text=capabilities_text, task_types=task_types)
    return result['choice']


def evaluate_candidate(task, subtasks, capabilities_input):
    prompt = f'''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}
    Given the parent task {task}, and its subtasks {subtasks}, 
    evaluate how well these subtasks address the requirements 
    of the parent task without any gaps or redundancies, using the following capabilities: 
    {capabilities_input}
    Return a score between 0.0 and 1.0, where 1.0 is the best possible score.
    
    Score:
    {{/user}}'''

    score = outlines_model(prompt=prompt, type="float")

    return score
