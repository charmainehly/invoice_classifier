'''
https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo
'''
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv('API_KEY')
print(openai.api_key)

def chat(prompt):
    """
    Function for generating a chat-based completion using the OpenAI API.

    Args:
        prompt (str): The user's message or prompt.

    Returns:
        str: The assistant's reply.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    reply = response['choices'][0]['message']['content']
    return reply