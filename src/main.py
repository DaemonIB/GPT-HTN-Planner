import threading

from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from gpt4_utils import get_initial_task, compress_capabilities
from htn_planner import htn_planning
from text_utils import trace_function_calls

app = Flask(__name__)
CORS(app)  # Add this line to enable CORS
socketio = SocketIO(app, cors_allowed_origins="*")

@trace_function_calls
def task_node_to_dict(task_node):
    if task_node is None:
        return None

    return {
        "task_name": task_node.task_name,
        "status": task_node.status,
        "children": [task_node_to_dict(child) for child in task_node.children]
    }

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def send_task_node_update(task_node):
    root_task_node = task_node
    while root_task_node.parent is not None:
        root_task_node = root_task_node.parent
    task_node_data = task_node_to_dict(root_task_node)
    socketio.emit('task_node_update', task_node_data)

def run_server():
    socketio.run(app, host="127.0.0.1", debug=True, use_reloader=False, port=5000, allow_unsafe_werkzeug=True, log_output=False)

def print_plan(task_node, depth=0):
    print(f"{'  ' * depth}- {task_node.task_name}")
    for child in task_node.children:
        print_plan(child, depth + 1)

def main():
    # Clear the log file at the beginning of each run
    with open('function_trace.log', 'w') as log_file:
        log_file.write("")

    initial_state_input = input("Describe the initial state: ")
    goal_input = input("Describe your goal: ")

    # Set default capabilities to a Linux terminal with internet access
    default_capabilities = "Linux terminal, internet access"
    print(f"Default capabilities: {default_capabilities}")
    capabilities_input = input("Describe the capabilities available (press Enter to use default): ")

    # Use default capabilities if the user doesn't provide any input
    if not capabilities_input.strip():
        capabilities_input = default_capabilities

    goal_task = get_initial_task(goal_input)
    compressed_capabilities = compress_capabilities(capabilities_input)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    print("Starting HTN planning with the initial goal task:", goal_task)
    plan = htn_planning(initial_state_input, goal_task, compressed_capabilities, 5, send_task_node_update)

    if plan:
        print("Plan found:")
        print_plan(plan)
    else:
        print("No plan found.")

if __name__ == '__main__':
    # Run the main function
    main()