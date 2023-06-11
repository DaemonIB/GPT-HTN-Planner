import guidance


def translate(original_task, capabilities_input, gpt4):
    # translates a task into a form that can be completed with the specified capabilities
    task_translation = guidance('''
    {{#system}}You are a helpful agent{{/system}}
    
    {{#user}}Translate the task '{{task}}' into a form that can be executed using the following capabilities:
    '{{capabilities_input}}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    
    When translated to use the specified capabilities the result is:{{/user}}
    {{#assistant}}{{gen "translated_task"}}{{/assistant}}
    ''', llm=gpt4)

    result = task_translation(task=original_task, capabilities_input=capabilities_input)
    return result['translated_task']


def is_task_primitive(task_name, capabilities_text, gpt4):
    primitive_check = guidance('''
    {{#system}}You are a helpful agent{{/system}}

    {{#user}}Given the task '{{task_name}}' and the capabilities '{{capabilities_text}}',
    determine if the task is primitive or compound.
    Please provide the answer as 'primitive' or 'compound':{{/user}}
    {{#assistant}}{{#select "choice"}} primitive{{or}} compound{{/select}}{{/assistant}}""")
    ''', llm=gpt4)

    result = primitive_check(task_name=task_name, capabilities_text=capabilities_text)
    return result['choice']
