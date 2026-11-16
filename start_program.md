# 🚀 Getting Started with the AI / GenAI Learning Project

Welcome! This guide outlines everything you need to set up, configure, and run each script in this learning repository.

---

## 📁 Project Structure Overview

```text
GEN_AI_LEARNING/
│
├── .env.example                               # Template for your API keys
├── requirements.txt                           # Python package dependencies
├── readme.md                                  # Original project notes
│
├── hello_world/
│   ├── gemini.py                              # Calls Google Gemini API using OpenAI client
│   └── main.py                                # Calls Groq API using OpenAI client
│
├── prompts/
│   ├── Cot.py                                 # Interactive Chain-of-Thought (math reasoning)
│   └── persona.py                             # Persona-based system prompt demo
│
├── trader/
│   └── watchlist.json                         # Watchlist data file
│
└── 19 - Running LLMs via Hugging Face Hub/
    └── readme.md                              # Notes for Hugging Face Hub & CLI
```

---

## 🛠️ Step 1: Prerequisites

Make sure you have **Python 3.10+** installed on your system.
Verify in your terminal:

```powershell
python --version
```

---

## 📦 Step 2: Create & Activate Virtual Environment

Creating an isolated virtual environment is recommended to avoid package conflicts.

1. Open PowerShell / Command Prompt in the project root directory (`e:\ai learning\GEN_AI_LEARNING`).
2. Run:

```powershell
python -m venv .venv
```

3. Activate the virtual environment:
   - **In PowerShell:**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **In Command Prompt (cmd):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
> **Tip (PowerShell restriction error?):**
> If PowerShell displays an error like `cannot be loaded because running scripts is disabled on this system`, run this once:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then retry activating the environment.

---

## 📥 Step 3: Install Required Dependencies

With the virtual environment activated (you will see `(.venv)` in your terminal prompt), install the dependencies:

```powershell
pip install -r requirements.txt
```

*(This installs `openai` and `python-dotenv`.)*

> **Optional (for Hugging Face experiments):**
> If you plan to run the experiments in `19 - Running LLMs via Hugging Face Hub`, also install:
> ```powershell
> pip install huggingface_hub transformers
> ```

---

## 🔑 Step 4: Set Up API Keys (`.env` File)

The scripts read API keys using `python-dotenv`.

1. Copy `.env.example` and name the new file `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Open the `.env` file in your editor and insert your actual keys:

```dotenv
# 1. Google Gemini API Key
# Get a free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY="your_actual_gemini_api_key_here"

# 2. Groq API Key (used in hello_world/main.py)
# Get a free key at: https://console.groq.com/keys
OPENAI_API_KEY="your_actual_groq_api_key_here"
```

> **Security Note:** `.env` is listed in `.gitignore` so your private API keys will never accidentally be committed to Git.

---

## ▶️ Step 5: How to Run the Programs

Run any of the Python scripts from the root directory:

### 1. Gemini Hello World
Tests the Gemini API with structured JSON output and system instructions.
```powershell
python hello_world/gemini.py
```

### 2. Groq / OpenAI Endpoint
Tests the Groq API endpoint with the OpenAI SDK.
```powershell
python hello_world/main.py
```

### 3. Interactive Chain-of-Thought (CoT) Prompt
An interactive script that asks for a math problem, plans step-by-step reasoning using Gemini, and prints the result.
```powershell
python prompts/Cot.py
```
*(Type your question when prompted, e.g. `What is 15 * 4 + 20 / 2?`)*

### 4. Persona Assistant
Prompts Gemini acting as an AI persona ("Parth Singh, Techie & Principal Engineer").
```powershell
python prompts/persona.py
```

---

## 💡 Troubleshooting & Notes

- **Model Name Errors (404 / 400 Bad Request):**
  If Gemini reports a model not found error, open the script and update the model identifier to a currently active model, such as:
  - `gemini-1.5-flash` or `gemini-2.0-flash`
- **Missing API Key:**
  If you get `NoneType` or authentication errors, make sure:
  1. The `.env` file is named exactly `.env` (not `.env.txt`).
  2. The `.env` file is placed in the root directory: `e:\ai learning\GEN_AI_LEARNING\.env`.
  3. You replaced the placeholder text with your real API key.
