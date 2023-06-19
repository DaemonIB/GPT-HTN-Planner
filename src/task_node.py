import uuid

from text_utils import trace_function_calls


class TaskNode:
    def __init__(self, task_name, parent=None, status=None):
        self.task_name = task_name
        self.node_name = str(uuid.uuid4())
        self.parent = parent
        self.children = []
        self.status = status

    @trace_function_calls
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    @trace_function_calls
    def update_task_name(self, task_name):
        self.task_name = task_name