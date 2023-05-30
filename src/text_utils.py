import os
import re
import datetime

def log_parsing_errors(input_text, extracted_list):
    log_dir = "../parsing_errors"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/parsing_errors.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Input text:\n{input_text}\n")
        log_file.write(f"{timestamp}: Extracted list:\n{', '.join(extracted_list)}\n\n")

def trace_function_calls(func):
    def wrapper(*args, **kwargs):
        with open('function_trace.log', 'a') as log_file:
            log_file.write(f"Function {func.__name__} called with arguments {args} and keyword arguments {kwargs}\n")
        result = func(*args, **kwargs)
        with open('function_trace.log', 'a') as log_file:
            log_file.write(f"Function {func.__name__} returned {result}\n")
        return result
    return wrapper

@trace_function_calls
def extract_lists(text):
    # Define patterns
    subtask_pattern = r'\[((?:[^\[\]]|\[[^\[\]]*\])*)\]'
    url_pattern = r'https?://\S+'
    command_pattern = r"(?<!\[)'(?:[^']|'')*?'(?!\])"
    backtick_command_pattern = r"`(.+?)`"

    # Replace URLs and commands with placeholders
    url_placeholder = "URL_PLACEHOLDER"
    command_placeholder = "COMMAND_PLACEHOLDER"
    urls_replacements = re.findall(url_pattern, text)
    commands_replacements = re.findall(command_pattern, text)
    text = re.sub(url_pattern, url_placeholder, text)
    text = re.sub(command_pattern, command_placeholder, text)

    # Extract subtasks
    subtask_list = re.findall(subtask_pattern, text)
    backtick_command_list = re.findall(backtick_command_pattern, text)

    # Combine lists
    combined_list = subtask_list + backtick_command_list

    # Clean list items by removing trailing punctuation marks and restoring URLs and commands
    cleaned_list = []
    for item in combined_list:
        cleaned_item = item.rstrip('.!?')

        # Check if the cleaned item is shorter than the original and ends with a word character
        # If so, append the last character from the original item
        if len(cleaned_item) < len(item) and re.search(r'\w$', item):
            cleaned_item += item[-1]

        # Restore URLs and commands
        while url_placeholder in cleaned_item and urls_replacements:
            cleaned_item = cleaned_item.replace(url_placeholder, urls_replacements.pop(0), 1)
        while command_placeholder in cleaned_item and commands_replacements:
            cleaned_item = cleaned_item.replace(command_placeholder, commands_replacements.pop(0), 1)

        cleaned_list.append(cleaned_item)

    return cleaned_list