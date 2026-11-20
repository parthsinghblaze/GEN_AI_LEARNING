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
import yfinance as yf

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

Available Tools:
- get_weather(city: str): Take city as an input string and return weather condition
- run_command(cmd: str): Takes a system linux command as string and executes the command on the user system and return the output of that command

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

Example3:

START: WHat is the stock price of reliance
PLAN: { "step": "PLAN", "content": "Seems like user is instrested in getting the live stock price of a reliance stock" }
PLAN: { "step": "PLAN", "content": "lets see if we have any avaiable took from the list of avaiable" },
PLAN: { "step": "PLAN", "content": "But user can misspelled the stock name as well so first i need to get the correct releiance stock symbol" },
PLAN: { "step": "PLAN", "content": "Suppose releiance get missspelled as relienaces like this tjhen we need to find the correct symblol" },
PLAN: { "step": "PLAN", "content": "So we need to find the current symbol of reliance if the user is misspelled" },
PLAN: { "step": "PLAN", "content": "Suppose user writes Yatra online ltd then we need to find the current symbol for this stock which is YATRA" },
PLAN: { "step": "PLAN", "content": "suppose user is sarcching for the kalyan jwellers tjem his symbol we need to find that is KALYANKJIL" },
PLAN: { "step": "PLAN", "content": "to get the stock symbol we need to find a tool" }
PLAN: { "step": "PLAN", "content": "Great, we have a get_stock_symbol tool to get the stock symbol" }
PLAN: { "step": "PLAN", "content": "I need to call get_stock_symbol tool for reveice the reliance stock symbol with the input reliance" }
PLAN: { "step": "TOOL", "tool": "get_stock_symbol", "input": "inox" }
PLAN: { "step": "OBSERVE", "tool": "get_stock_symbol", "output": "{'INOXWIND', 'INOXINDIA', 'PVRINOX', 'INOXGREEN'}" }
PLAN: { "step": "PLAN", "content": "Great, we have get_stock_price tool abaiable from this query"}
PLAN: { "step": "PLAN", "content": "I need to call get_stock_price tiik for reliance stock" },
PLAN: { "step": "PLAN", "content": "After gettig the reliance symbol we will call the get_stock_price tool for the RELIANCE stock and reliance as an input " }
PLAN: { "step": "TOOL", "tool": "get_stock_price", "input": "RELIANCE" },
PLAN: { "step": "OBSERVE", "tool" : "get_stock_price", "output": "Stock price of NSE in nse is 2120 found and bse is 2129.6"},
PLAN: { "step": "PLAN", "content": "Great I got the stock info about the Reliance" },
OUTPUT: { "step": "OUTPUT", "content": "The current price of reliance in NSE is 2120 and in BSE is 2121" }
"""

def run_command(cmd: str):
    result = os.system(cmd)
    return result


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"

    return "Something went wrong"


def get_stock_symbol(stockName: str):
    """
    Finds the official ticker symbol for a given company name or typo.
    Example: 'tata moters' -> 'TATAMOTORS'
    """
    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"q": stockName}
    all_stock_symbol = set()

    try:
        res = requests.get(url, params=params, headers=headers).json()
        quotes = res.get("quotes", [])

        # Look for Indian exchange symbols ending in .NS or .BO    
        for q in quotes:
            symbol = q.get("symbol", "")
            print(f"symbol => {symbol}")
            if symbol.endswith(".NS") or symbol.endswith(".BO"):
               all_stock_symbol.add(symbol.rsplit(".", 1)[0])
        
    except Exception:
        pass

    return sorted(all_stock_symbol)

# print(get_stock_symbol("inox"))

def get_stock_price(stockName: str):
    stock = stockName.strip().upper()
    # Get NSE Price
    try:
        nse_price = round(yf.Ticker(f"{stock}.NS").fast_info['last_price'], 2)
    except Exception:
        nse_price = "not found"
    # Get BSE Price
    try:
        bse_price = round(yf.Ticker(f"{stock}.BO").fast_info['last_price'], 2)
    except Exception:
        bse_price = "not found"
    # If neither is found, return null (None)
    if nse_price == "not found" and bse_price == "not found":
        return None
    return f"Stock price of {stock} in nse is {nse_price} and bse is {bse_price}"

avaiable_tool = {
    "get_weather": get_weather,
    "get_stock_price": get_stock_price,
    "get_stock_symbol": get_stock_symbol,
    "run_command": run_command
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

while True:
    main()


# https://wttr.in/daman