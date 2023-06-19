# An implementation of HTN using the GPT-4 API
# Due to the expressiveness of language, a lot of steps that would generally require complex functions are left up
# to the LLM

from gpt4_utils import gpt4_is_goal, is_task_primitive, can_execute, log_state_change
from openai_api import call_openai_api, log_response
from task_node import TaskNode
from text_utils import extract_lists, trace_function_calls
from guidance_prompts import htn_prompts

class HTNPlanner:
    def __init__(self, initial_state, goal_task, capabilities_input, max_depth=5, send_update_callback=None):
        self.initial_state = initial_state
        self.goal_task = goal_task
        self.capabilities_input = capabilities_input
        self.max_depth = max_depth
        self.send_update_callback = send_update_callback

    def htn_planning(self):
        # Storage for successful task_node's so that they don't need to get regenerated for similar inputs
        db = None  # db = VectorDB()
        root_node = TaskNode(self.goal_task)
        while self.replan_required(self.initial_state, self.goal_task, root_node):
            root_node = self.htn_planning_recursive(
                self.initial_state,
                self.goal_task,
                root_node,
                self.max_depth,
                self.capabilities_input,
                db,
                self.send_update_callback,
            )
        return root_node

    @trace_function_calls
    def htn_planning_recursive(self, state, goal_task, root_node, max_depth, capabilities_input, db, send_update_callback=None):
        if gpt4_is_goal(state, goal_task):
            return root_node

        if send_update_callback:
            send_update_callback(root_node)

        success, updated_state = self.decompose(root_node, state, 0, max_depth, capabilities_input, goal_task,
                                        db, send_update_callback)
        if success:
            root_node.status = "succeeded"
            state = updated_state
            return root_node
        else:
            root_node.status = "failed"

        return root_node

    @trace_function_calls
    def replan_required(self, state, goal_task, task_node):
        if gpt4_is_goal(state, goal_task):
            return False
        if task_node is None or task_node.children == []:
            return True
        return False


    @trace_function_calls
    def translate_task(self, task, capabilities_input):
        response = htn_prompts.translate(task, capabilities_input)
        translated_task = response.strip()
        log_response("translate_task", translated_task)
        return translated_task


    # Add a new function to check if subtasks meet the requirements
    @trace_function_calls
    def check_subtasks(self, task, subtasks, capabilities_input):
        result = htn_prompts.check_subtasks(task, subtasks, capabilities_input)
        log_response("check_subtasks", result)
        return result == 'true'


    @trace_function_calls
    def decompose(self, task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback=None,
                n_candidates=3):
        task = task_node.task_name
        decompose_state = state

        # similar_task_nodes = db.query_by_name(task_node.task_name)
        #
        # if similar_task_nodes != None and similar_task_nodes != []:
        #     task_node = similar_task_nodes[0]

        if depth > max_depth:
            return False, decompose_state

        remaining_decompositions = max_depth - depth
        # When reaching the maximum depth, we will assume that the task is good due to the check_subtasks clearing it before
        # this point in the code is reached, we only need to know if its primitive or not if we intend to return early
        if remaining_decompositions == 0:
            return True, decompose_state
        else:
            if is_task_primitive(task, capabilities_input):
                # Translate the task before checking if it can be executed
                translated_task = self.translate_task(task, capabilities_input)

                # Needs pre-conditions to prevent discontinuities in the graph
                if can_execute(translated_task, capabilities_input, decompose_state):
                    task_node.update_task_name(translated_task)  # Update the task with the translated form
                    print(f"Executing task:\n{translated_task}")
                    updated_state = self.execute_task(state, translated_task)
                    decompose_state = updated_state
                    return True, decompose_state
                else:
                    return False, decompose_state
            else:
                print(f"Decomposing task:\n{task}")

                success = False
                best_candidate = None  # Add a variable to store the best candidate
                best_candidate_score = float('-inf')  # Add a variable to store the best candidate score

                """
                Create n candidate lists of subtask decompositions. If one list of subtasks passes
                the check_subtasks requirements then continue using that candidate.
                If the list of subtasks fails the check then continue until one passes or candidates are exhausted.
                Track the effectiveness of each candidate using the evaluate_candidate function
                If candidates are exhausted, choose the best candidate list of subtasks and continue.
                """
                candidates = []
                subtasks_list = []
                for _ in range(n_candidates):
                    subtasks_list = self.get_subtasks(task, decompose_state, remaining_decompositions, capabilities_input)
                    score = self.evaluate_candidate(task, [subtask for subtask in subtasks_list], capabilities_input)
                    candidates.append((subtasks_list, score))

                # Sort candidates by their score
                candidates.sort(key=lambda x: x[1], reverse=True)

                for subtasks_list, score in candidates:
                    if self.check_subtasks(task, [subtask for subtask in subtasks_list], capabilities_input):
                        print(f"Successfully decomposed task into subtasks:\n'{', '.join(subtasks_list)}'")
                        success = True
                        break

                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = subtasks_list

                if not success and best_candidate is not None:
                    print(f"No candidates met the requirements, using the best candidate:\n'{', '.join(best_candidate)}'")
                    subtasks_list = best_candidate

                if success or best_candidate is not None:
                    for subtask in subtasks_list:
                        if send_update_callback:  # Add this line
                            send_update_callback(task_node)  # Add this line

                        subtask_node = TaskNode(subtask, parent=task_node)
                        task_node.add_child(subtask_node)

                        success, updated_state = self.decompose(subtask_node, decompose_state, depth + 1, max_depth,
                                                        capabilities_input,
                                                        goal_state, db, send_update_callback)

                        if success:
                            decompose_state = updated_state
                            task_node.status = "succeeded"

                            if send_update_callback:
                                send_update_callback(task_node)
                        else:
                            task_node.status = "failed"
                            task_node.children.clear()
                            break

                # Update the db with the current task_node
                # if success:
                #     db.add_task_node(task_node)

                return success, decompose_state


    @trace_function_calls
    def evaluate_candidate(self, task, subtasks, capabilities_input):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            # Max 10 token or 8 digits after the decimal 0.99999999
            response = htn_prompts.evaluate_candidate(task, subtasks, capabilities_input)
            try:
                score = float(response.strip())
                log_response("evaluate_candidate", score)
                return score
            except ValueError:
                retries += 1
                if retries >= max_retries:
                    raise ValueError("Failed to convert response to float after multiple retries.")


    @trace_function_calls
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input):
        subtasks_with_types = htn_prompts.get_subtasks(task, state, remaining_decompositions, capabilities_input)
        print(f"Decomposing task {task} into candidates:\n{subtasks_with_types}")
        subtasks_list = extract_lists(subtasks_with_types)
        return subtasks_list


    # Update the execute_task function to log state changes
    @trace_function_calls
    def execute_task(self, state, task):
        prompt = (f"Given the current state '{state}' and the task '{task}', "
                f"update the state after executing the task:")

        response = call_openai_api(prompt)

        updated_state = response.choices[0].message.content.strip()
        log_response("execute_task", task)
        log_state_change(state, updated_state, task)  # Add this line to log state changes
        return updated_state
