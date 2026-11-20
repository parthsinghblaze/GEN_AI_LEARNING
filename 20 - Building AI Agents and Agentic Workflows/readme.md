# 1. Agentic AI Fundamentals — Section Intro

### The Evolution: From Passive Chatbots to Active Agents
The AI landscape is currently undergoing a massive paradigm shift:

- **Phase 1: Generative AI (Chatbots & Advisory)**
  - *What it did:* Users asked questions, and the model gave advice, code, or summaries.
  - *Limitation:* The AI was passive. A human still had to copy-paste the code, query the database, or click the buttons.
  
- **Phase 2: Agentic AI (Action & Automation)**
  - *What it does:* Instead of just talking about the task, the AI **executes** it end-to-end.
  - *Capability:* It can browse the web, write and run code, trigger database transactions, handle edge cases, and self-correct when things go wrong.

### Why Is Agentic AI the Next Big Thing?
1. **From Answering to Doing:** Moving from *"Here is how to cancel an order"* to *"I have cancelled the order and refunded the customer."*
2. **Goal-Driven Autonomy:** You provide a high-level goal (e.g., *"Find the lowest flight price for next Monday and book it"*), and the agent breaks it down into sub-steps without needing constant step-by-step human prompts.
3. **Tool Integration:** It bridges the gap between raw LLM intelligence and real-world software systems (APIs, databases, internal tools).

## 2. What Exactly Are AI Agents?

### The Traditional Customer Support Flow:
- **User -> Human Support Agent -> Company Server/Database**
- A human customer support agent has access to company dashboards, order history, cart details, and shipping status. When a customer says *"Please cancel my order,"* the human verifies the details and clicks the button / runs the action to update the database.

### The Limitation of Standalone LLMs:
- Models like OpenAI GPT or Google Gemini are **stateless reasoning engines**.
- By default, they are passive: they take **text in** and return **text out**.
- They have **no direct access** to your private database, APIs, payment gateways, or real-time business state.

### Bridging the Gap: What Makes It an "AI Agent"?
An **AI Agent** is not just an LLM. It is an LLM wrapped in software that provides:
1. **Reasoning (The Brain):** The LLM understands the customer's intent and decides what step to take next.
2. **Tools & Function Calling (The Hands):** Pre-defined APIs exposed by developers (e.g., `fetch_order()`, `cancel_order()`, `process_refund()`).
3. **Memory:** Short-term conversation history and long-term user profile data.
4. **Autonomous Action Loop:** The agent observes the user request -> decides which tool to trigger -> observes the API response -> formulates a final reply to the user.

> **Key Takeaway:** An LLM generates text. An **AI Agent** uses the LLM's reasoning capability to make decisions, invoke tools, interact with databases, and complete tasks autonomously.

## 3. Coding Your First AI Agent: Weather Agent

In this hands-on project, we build a **Weather AI Agent** from scratch using the **ReAct (Reason + Act)** pattern. 

Instead of relying on built-in tool calling frameworks, we implement the agentic loop manually so we can understand exactly how an LLM:
1. **Plans** its thoughts step-by-step (`PLAN`).
2. **Decides** when an external tool is required (`TOOL`).
3. **Observes** the output returned by the external tool (`OBSERVE`).
4. **Formulates** the final user-facing response (`OUTPUT`).

---

### Step-by-Step Breakdown

#### Step 1: Environment Setup & Client Initialization
We import the necessary libraries (`openai`, `dotenv`, `requests`, `json`) and initialize the OpenAI client to connect to Google's Gemini endpoint using `GEMINI_API_KEY`.

```python
import os
import json
from openai import (
    OpenAI,
    BadRequestError,
    RateLimitError,
    APIStatusError
)
from dotenv import load_dotenv
import requests
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

- **`requests`:** Used to make HTTP calls to the weather API.
- **`load_dotenv()`:** Reads the API key from our `.env` file.
- **`client`:** Configured with Google's OpenAI-compatible base URL.

---

#### Step 2: System Prompt & ReAct Protocol Definition
The system prompt teaches the LLM how to think and behave like an agent. It enforces a strict JSON schema and provides few-shot examples of how to plan, request tools, and produce output.

```python
SYSTEM_PROMPT = """
 You're an expert AI Assistant in resolving user queries using chain of thought.
 You work on START, PLAN and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be multiple steps.
 Once you think enough PLAN han been done, finally you can give an OUTPUT.
 You can also call a tool if required from the list of avaiable tools
 for every tool call wait for the observe step which is the output from the called tool.

 Rules:
 - Strickly Follow the given JSON output format.
 - Only run one step at a time.
 - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to the displayed to the user)

Output JSON format:
{
    "step": "START" | "PLAN" | "OUTPUT" | "TOOL",
    "content": "string",
    "tool": "string",
    "input": "string"
}

Example1: 
START: Hey, Can you solve 2+2*5 / 10
PLAN: {"step": "PLAN", content: "seems like user is intereted in math promblem"}
PLAN: {"step": "PLAN", "content": "Looking at the promblem, we should solve this using BODMAN method"}
PLAN: {"step": "PLAN", "content": "first we should solve the bracket 2+2 = 4"}
PLAN: {"step": "PLAN", "content": "then solve division 5 / 10 = 0.5"}
PLAN: {"step": "PLAN", "content": "then solve addition 4 + 0.5 = 4.5"}
PLAN: {"step": "PLAN", "content": "final answer is 4.5"}
OUTPUT: {"step": "OUTPUT", "content": "The answer to 2+2*5 / 10 is 4.5"}

Example2:

START: What is the weather of Delhi?
PLAN: { "step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi" }
PLAN: { "step": "PLAN", "content": "Lets see if we have any avaiable tool from the list of avaiable" }
PLAN: { "step": "PLAN", "content": "Great, we have get_weather tool avaible from this query." }
PLAN: { "step": "PLAN", "content": "I need to call get_weather tool for delhi as input for city" }
PLAN: { "step": "TOOL", "tool": "get_weather", "input": "delhi" }
PLAN: { "step": "OBSERVE", "tool" : "get_weather", "output": "them temp of delhi is cloudy with 20 C"}
PLAN: { "step": "PLAN", "content": "Great I got the weather info about the delhi" },
OUTPUT: { "step": "OUTPUT", "content": "The current weather of delhi is 20 C with with some cloudy sky" }
"""
```

- **`response_format={"type": "json_object"}`:** Ensures the model only speaks in structured JSON.
- **Rules & Examples:** Provide the model with few-shot demonstration of the reasoning and tool execution sequence.

---

#### Step 3: Tool Function & Available Tools Registry
We create our real-world tool function `get_weather(city)` that queries `wttr.in`, and store it inside a dictionary (`avaiable_tool`) so the agent can look up and run tools dynamically by name.

```python
def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"

avaiable_tool = {
    "get_weather": get_weather
}
```

---

#### Step 4: The Autonomous Execution Loop (`main`)
The `while True` loop is the heart of the agent:
1. It sends the current conversation history to the LLM.
2. It parses the JSON response and checks the `step` key:
   - If `START`: Prints the start content and continues.
   - If `PLAN`: Prints the thought/reasoning, and sends a `"Continue with the next step."` prompt back to the model.
   - If `TOOL`: Looks up the tool in `avaiable_tool`, executes it with the provided input, and appends the result as an `OBSERVE` step into the message history.
   - If `OUTPUT`: Prints the final result to the user and breaks out of the loop.

```python
def main():
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    user_query = input("> ")
    
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
                print(f"parsed_result {parsed_result}")
                if parsed_result.get("step") == "START":
                    print("🔥", parsed_result.get("content"))
                    continue

                if parsed_result.get("step") == "TOOL":
                    tool_to_call = parsed_result.get("tool")
                    tool_input = parsed_result.get("input")
                    print(f"👀: {tool_to_call} ({tool_input})")
                    tool_response = avaiable_tool[tool_to_call](tool_input)

                    message_history.append({
                        "role": "user",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_to_call,
                                "input": tool_input,
                                "output": tool_response
                            }
                        )
                    })
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
```

---

#### Step 5: Robust Exception Handling
Catches standard OpenAI and Gemini API errors such as rate limits, bad requests, or API status issues so the agent fails gracefully.

```python
    except RateLimitError as e:
        print("Rate limit / quota exceeded!")
        print("Details:", str(e))

    except BadRequestError as e:
        print("Bad Request:", str(e))

    except APIStatusError as e:
        print("API Status:", e.status_code)
        print("Details:", str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()
```

---

### Whole Complete Code

Here is the entire program combined:

```python
import os
import json
from openai import (
    OpenAI,
    BadRequestError,
    RateLimitError,
    APIStatusError
)
from dotenv import load_dotenv
import requests
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


SYSTEM_PROMPT = """
 You're an expert AI Assistant in resolving user queries using chain of thought.
 You work on START, PLAN and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be multiple steps.
 Once you think enough PLAN han been done, finally you can give an OUTPUT.
 You can also call a tool if required from the list of avaiable tools
 for every tool call wait for the observe step which is the output from the called tool.

 Rules:
 - Strickly Follow the given JSON output format.
 - Only run one step at a time.
 - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to the displayed to the user)

Output JSON format:
{
    "step": "START" | "PLAN" | "OUTPUT" | "TOOL",
    "content": "string",
    "tool": "string",
    "input": "string"
}

Example1: 
START: Hey, Can you solve 2+2*5 / 10
PLAN: {"step": "PLAN", content: "seems like user is intereted in math promblem"}
PLAN: {"step": "PLAN", "content": "Looking at the promblem, we should solve this using BODMAN method"}
PLAN: {"step": "PLAN", "content": "first we should solve the bracket 2+2 = 4"}
PLAN: {"step": "PLAN", "content": "then solve division 5 / 10 = 0.5"}
PLAN: {"step": "PLAN", "content": "then solve addition 4 + 0.5 = 4.5"}
PLAN: {"step": "PLAN", "content": "final answer is 4.5"}
OUTPUT: {"step": "OUTPUT", "content": "The answer to 2+2*5 / 10 is 4.5"}

Example2:

START: What is the weather of Delhi?
PLAN: { "step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi" }
PLAN: { "step": "PLAN", "content": "Lets see if we have any avaiable tool from the list of avaiable" }
PLAN: { "step": "PLAN", "content": "Great, we have get_weather tool avaible from this query." }
PLAN: { "step": "PLAN", "content": "I need to call get_weather tool for delhi as input for city" }
PLAN: { "step": "TOOL", "tool": "get_weather", "input": "delhi" }
PLAN: { "step": "OBSERVE", "tool" : "get_weather", "output": "them temp of delhi is cloudy with 20 C"}
PLAN: { "step": "PLAN", "content": "Great I got the weather info about the delhi" },
OUTPUT: { "step": "OUTPUT", "content": "The current weather of delhi is 20 C with with some cloudy sky" }
"""

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"

avaiable_tool = {
    "get_weather": get_weather
}

def main():
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    user_query = input("> ")
    
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
                print(f"parsed_result {parsed_result}")
                if parsed_result.get("step") == "START":
                    print("🔥", parsed_result.get("content"))
                    continue

                if parsed_result.get("step") == "TOOL":
                    tool_to_call = parsed_result.get("tool")
                    tool_input = parsed_result.get("input")
                    print(f"👀: {tool_to_call} ({tool_input})")
                    tool_response = avaiable_tool[tool_to_call](tool_input)

                    message_history.append({
                        "role": "user",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_to_call,
                                "input": tool_input,
                                "output": tool_response
                            }
                        )
                    })
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
    
    except RateLimitError as e:
        print("Rate limit / quota exceeded!")
        print("Details:", str(e))

    except BadRequestError as e:
        print("Bad Request:", str(e))

    except APIStatusError as e:
        print("API Status:", e.status_code)
        print("Details:", str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()

main()


# https://wttr.in/daman
```

---

## 4. Enforcing Structured Output with Pydantic

### The Problem with Simple JSON Mode (`json_object`)
In Section 3, we used `response_format={"type": "json_object"}`. 

While this ensures that the model outputs valid JSON syntax (no plain conversational text or unclosed brackets), **it does not guarantee the schema**. The LLM can still:
- Miss required keys (e.g., forget `"step"`).
- Hallucinate key names (e.g., output `"action"` instead of `"tool"`).
- Return unexpected data types (e.g., an array instead of a string).

When using `json.loads()`, any unexpected structure forces you to write defensive code with lots of `.get()` calls or risk your agent crashing at runtime.

---

### The Solution: Pydantic + Strict JSON Schema
We use **Pydantic** models combined with OpenAI/Gemini's **Structured Outputs** (`strict: True`). 

This mathematically constrains the model's token sampling to **only** generate tokens that match our exact Pydantic schema, guaranteeing:
1. All required fields are present.
2. Field types strictly match.
3. We get full Python type hinting and dot notation (`parsed_result.step`) instead of dictionary `.get()`.

---

### Step-by-Step Code Changes

#### Change 1: Added Pydantic Imports
```python
from pydantic import BaseModel, Field
from typing import Optional
```

#### Change 2: Defined the Pydantic Schema (`MyOutputFormat`)
We define a schema describing all potential fields the agent might return during `PLAN`, `TOOL`, or `OUTPUT` steps:

```python
class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="the optional string content for the step")
    tool: Optional[str] = Field(None, description="the ID of the tool to call")
    input: Optional[str] = Field(None, description="the input params for the tool")
```

- `Field(...)`: Marks `step` as strictly required.
- `Optional[str]`: Marks `content`, `tool`, and `input` as optional because not every step uses all fields (e.g., `PLAN` uses `content`, while `TOOL` uses `tool` and `input`).

#### Change 3: Enforcing Strict JSON Schema via `client.beta.chat.completions.parse`
Instead of standard JSON mode, we pass the generated JSON schema and set `"strict": True`:

```python
response = client.beta.chat.completions.parse(
    model="gemini-3.5-flash-lite",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "MyOutputFormat",
            "schema": MyOutputFormat.model_json_schema(),
            "strict": True,
        },
    },
    messages=message_history
)
```

#### Change 4: Type-Safe Validation & Dot Notation Access
Instead of `json.loads(raw_result)` and `parsed_result.get("step")`, we validate directly with Pydantic:

```python
# Validate the JSON into a strongly-typed Pydantic object
parsed_result = MyOutputFormat.model_validate_json(raw_result)

# Access fields directly with clean dot notation:
if parsed_result.step == "START":
    print("🔥", parsed_result.content)
    continue

if parsed_result.step == "TOOL":
    tool_to_call = parsed_result.tool
    tool_input = parsed_result.input
    ...

if parsed_result.step == "PLAN":
    print("🧠", parsed_result.content)
    ...

if parsed_result.step == "OUTPUT":
    print("👍", parsed_result.content)
    break
```

---

### Whole Complete Code with Pydantic Enforcement

```python
import os
import json
from openai import (
    OpenAI,
    BadRequestError,
    RateLimitError,
    APIStatusError
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional
import requests
load_dotenv()

print(os.getenv("GEMINI_API_KEY"))
client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


SYSTEM_PROMPT = """
 You're an expert AI Assistant in resolving user queries using chain of thought.
 You work on START, PLAN and OUTPUT steps.
 You need to first PLAN what needs to be done. The PLAN can be multiple steps.
 Once you think enough PLAN han been done, finally you can give an OUTPUT.
 You can also call a tool if required from the list of avaiable tools
 for every tool call wait for the observe step which is the output from the called tool.

 Rules:
 - Strickly Follow the given JSON output format.
 - Only run one step at a time.
 - The sequence of steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to the displayed to the user)

Output JSON format:
{
    "step": "START" | "PLAN" | "OUTPUT" | "TOOL",
    "content": "string",
    "tool": "string",
    "input": "string"
}

Example1: 
START: Hey, Can you solve 2+2*5 / 10
PLAN: {"step": "PLAN", content: "seems like user is intereted in math promblem"}
PLAN: {"step": "PLAN", "content": "Looking at the promblem, we should solve this using BODMAN method"}
PLAN: {"step": "PLAN", "content": "first we should solve the bracket 2+2 = 4"}
PLAN: {"step": "PLAN", "content": "then solve division 5 / 10 = 0.5"}
PLAN: {"step": "PLAN", "content": "then solve addition 4 + 0.5 = 4.5"}
PLAN: {"step": "PLAN", "content": "final answer is 4.5"}
OUTPUT: {"step": "OUTPUT", "content": "The answer to 2+2*5 / 10 is 4.5"}

Example2:

START: What is the weather of Delhi?
PLAN: { "step": "PLAN", "content": "Seems like user is interested in getting weather of Delhi" }
PLAN: { "step": "PLAN", "content": "Lets see if we have any avaiable tool from the list of avaiable" }
PLAN: { "step": "PLAN", "content": "Great, we have get_weather tool avaible from this query." }
PLAN: { "step": "PLAN", "content": "I need to call get_weather tool for delhi as input for city" }
PLAN: { "step": "TOOL", "tool": "get_weather", "input": "delhi" }
PLAN: { "step": "OBSERVE", "tool" : "get_weather", "output": "them temp of delhi is cloudy with 20 C"}
PLAN: { "step": "PLAN", "content": "Great I got the weather info about the delhi" },
OUTPUT: { "step": "OUTPUT", "content": "The current weather of delhi is 20 C with with some cloudy sky" }
"""

def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"

avaiable_tool = {
    "get_weather": get_weather
}

class MyOutputFormat(BaseModel):
    step: str = Field(..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc")
    content: Optional[str] = Field(None, description="the optional string content for the step")
    tool: Optional[str] = Field(None, description="the ID of the tool to call")
    input: Optional[str] = Field(None, description="the input params for the tool")

def main():
    message_history = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    user_query = input("> ")
    message_history.append({"role": "user", "content": user_query})

    try:
        while True:
                response = client.beta.chat.completions.parse(
                    model="gemini-3.5-flash-lite",
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "MyOutputFormat",
                            "schema": MyOutputFormat.model_json_schema(),
                            "strict": True,
                        },
                    },
                    messages=message_history
                )
            
                raw_result = response.choices[0].message.content
                message_history.append({"role": "assistant", "content": raw_result})
                parsed_result = MyOutputFormat.model_validate_json(raw_result)

                print(f"parsed_result {parsed_result}")
                if parsed_result.step == "START":
                    print("🔥", parsed_result.content)
                    continue

                if parsed_result.step == "TOOL":
                    tool_to_call = parsed_result.tool
                    tool_input = parsed_result.input
                    print(f"👀: {tool_to_call} ({tool_input})")
                    tool_response = avaiable_tool[tool_to_call](tool_input)

                    message_history.append({
                        "role": "user",
                        "content": json.dumps(
                            {
                                "step": "OBSERVE",
                                "tool": tool_to_call,
                                "input": tool_input,
                                "output": tool_response
                            }
                        )
                    })
                    continue

                if parsed_result.step == "PLAN":
                    print("🧠", parsed_result.content)
                    message_history.append({
                        "role": "user",
                        "content": "Continue with the next step."
                    })
                    continue

                if parsed_result.step == "OUTPUT":
                    print("👍", parsed_result.content)
                break
    
    except RateLimitError as e:
        print("Rate limit / quota exceeded!")
        print("Details:", str(e))

    except BadRequestError as e:
        print("Bad Request:", str(e))

    except APIStatusError as e:
        print("API Status:", e.status_code)
        print("Details:", str(e))

    except Exception as e:
        import traceback
        traceback.print_exc()

main()


# https://wttr.in/daman
```
5. Building a cli coding agent from scratch


def run_command(cmd: str):
    result = os.system(cmd)
    return result

Added the tool in the object
avaiable_tool = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "get_stock_symbol": get_stock_symbol,
    "run_command": run_command
}

