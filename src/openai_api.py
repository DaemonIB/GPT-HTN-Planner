import datetime
import os
import time

import openai
from ratelimiter import RateLimiter
import guidance

openai.api_key = os.environ.get('OPENAI_KEY')

# Configure the rate limiter to allow a maximum of 10 calls per minute
rate_limiter = RateLimiter(max_calls=10, period=60)


def call_openai_api(prompt, max_tokens=None, temperature=1.0, strip=False):
    retries = 3
    delay = 5

    while retries > 0:
        try:
            # Use the rate limiter to ensure the API is not called too frequently
            with rate_limiter:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                    temperature=temperature,
                )
            return response.choices[0].message.content.strip() if strip else response
        except openai.error.RateLimitError as e:
            print(f"RateLimitError encountered: {e}. Retrying in {delay} seconds...")
            retries -= 1
            time.sleep(delay)
        except openai.error.APIError as e:
            print(f"APIError encountered: {e}. Retrying in {delay} seconds...")
            retries -= 1
            time.sleep(delay)

    raise Exception("Failed to get a response from the GPT-4 API after multiple retries.")


updated_log_files = {}


def log_response(function_name, response):
    global updated_log_files

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/{function_name}.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        if function_name not in updated_log_files:
            log_file.write("\n--- Application run start ---\n")
            updated_log_files[function_name] = True
        log_file.write(f"{timestamp}:\n{response}\n")
