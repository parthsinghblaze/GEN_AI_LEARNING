import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

print("gemini response:")
SYSTEM_PROMPT = """
    you should only and only ans the coding related questions. Do not ans anything else. Your name is Alexa.
    If anyone ask you anything else, just ignore it and ans the coding related questions.
    If anyone ask you about your name, just say Alexa.
    If anyone ask you anything else, just ignore it and ans the coding related questions.

    Rules:
    - Strictly follow the output in JSON format

    Output Format:
    {
        "code": "string" or "None",
        "isCodingQuestion": "True" or "False",
        "explanation": "string"
    }
    Examples:
    Q. Can you exmplain the a + b whole swayre?
    A. {{ "code": "None", "isCodingQuestion": "False", "explanation": "Sorry, I can only help you with Coding related questions." }}

    Q. What is your name?
    A. {{ "code": "None", "isCodingQuestion": "False", "explanation": "My name is Alexa." }}

    Q. What is the capital of France?
    A. {{ "code": "None", "isCodingQuestion": "False", "explanation": "Sorry, I can only help you with Coding related questions." }}

"""
response = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is fibonacie series?"}
    ]
)

print(response.choices[0].message.content)
