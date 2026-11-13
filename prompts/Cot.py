import os
import json
from openai import BadRequestError, OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

print("gemini response:")
SYSTEM_PROMPT = """
 You're an expert AI Assistant in resolving user queries using chain of thought.
 You work on START, PLAN and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be multiple steps.
 Once you think enough PLAN han been done, finally you can give an OUTPUT.

 Rules:
 - Strickly Follow the given JSON output format.
 - Only run one step at a time.
 - The sequence of steps is START (where yser gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to the displayed to the user)

Output JSON format:
{
    "step": "START" | "PLAN" | "OUTPUT",
    "content": "string"
}

Example:

START: Hey, Can you solve 2+2*5 / 10
PLAN: {"step": "PLAN", content: "seems like user is intereted in math promblem"}

PLAN: {"step": "PLAN", "content": "Looking at the promblem, we should solve this using BODMAN method"}

PLAN: {"step": "PLAN", "content": "first we should solve the bracket 2+2 = 4"}

PLAN: {"step": "PLAN", "content": "then solve division 5 / 10 = 0.5"}

PLAN: {"step": "PLAN", "content": "then solve addition 4 + 0.5 = 4.5"}

PLAN: {"step": "PLAN", "content": "final answer is 4.5"}

OUTPUT: {"step": "OUTPUT", "content": "The answer to 2+2*5 / 10 is 4.5"}

"""

message_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

user_query = input("Please write your math promblem: ")
message_history.append({"role": "user", "content": user_query})
try:
    while True:
        response = client.chat.completions.create(
            model="gemini-3.5-flash-lite",
            response_format={"type": "json_object"},
            messages=message_history
        )

        raw_result = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": raw_result})
        parsed_result = json.loads(raw_result)

        if parsed_result.get("step") == "START":
            print("🔥", parsed_result.get("content"))
            continue

        if parsed_result.get("step") == "PLAN":
            print("🧠", parsed_result.get("content"))
            message_history.append({
                "role": "user",
                "content": "Continue with the next step."
            })
            continue

        if parsed_result.get("step") == "OUTPUT":
            print("👍", parsed_result.get("content"))
            break

except BadRequestError as e:
    print("Status Code:", e.status_code)
    print("Error Details:", e.response.text)
