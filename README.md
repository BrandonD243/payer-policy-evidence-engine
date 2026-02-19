<<<<<<< HEAD
# Payer Policy Reasoning Engine

A knowledge-graph–driven system for evaluating medical prior authorization
decisions (starting with MRI Lumbar Spine).

## Architecture Overview

This project separates concerns into three layers:

1. **Knowledge Modeling**
   - Universal clinical concepts and archetypes
   - Policy-specific clauses and criteria
   - Compiled into static JSON knowledge graphs

2. **Runtime Compilation**
   - Patient documents are extracted into structured facts
   - Only relevant policy and KG slices are compiled per case

3. **Agentic Reasoning**
   - A decision agent evaluates approval/denial
   - Outputs are validated against expected outcomes

## Key Directories
Payer Policy Reasoning Engine

A knowledge-graph–driven system for evaluating medical prior authorization
decisions (starting with MRI Lumbar Spine).

Architecture Overview

This project separates concerns into three layers:

Knowledge Modeling

Universal clinical concepts and archetypes

Policy-specific clauses and criteria

Compiled into static JSON knowledge graphs

Runtime Compilation

Patient documents are extracted into structured facts

Only relevant policy and KG slices are compiled per case

Agentic Reasoning

A decision agent evaluates approval/denial

Outputs are validated against expected outcomes

Key Directories
api/                  → FastAPI application
concepts/             → Canonical clinical concept registry
evaluators/           → Clause evaluation logic
examples/             → Sample patient cases
extractors/           → LLM-based concept extraction
payers/               → Policy clause definitions
storage/              → Case and document persistence

🚀 Local Setup (macOS / Apple Laptop)

These instructions assume no prior Python setup.

1️⃣ Install Python (if not already installed)

Open Terminal and check:

python3 --version


If Python 3.10+ is installed, you're good.

If not, install it using Homebrew:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python


Then verify:

python3 --version

2️⃣ Clone the Repository

In Terminal:

git clone https://github.com/YOUR_USERNAME/payer-policy-evidence-engine.git
cd payer-policy-evidence-engine


Replace YOUR_USERNAME with your GitHub username.

3️⃣ Create a Virtual Environment
python3 -m venv venv


Activate it:

source venv/bin/activate


Your terminal should now show (venv) at the beginning of the line.

4️⃣ Install Dependencies

If requirements.txt exists:

pip install -r requirements.txt


If not:

pip install fastapi uvicorn pydantic openai pyyaml

5️⃣ Set Environment Variables (OpenAI Key)

If your extractor uses OpenAI, you must set your API key:

export OPENAI_API_KEY="your_api_key_here"


To make it permanent:

echo 'export OPENAI_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc

6️⃣ Run the Backend

Start the FastAPI server:

uvicorn api.approval_api:app --reload


You should see:

Uvicorn running on http://127.0.0.1:8000


Open in your browser:

http://127.0.0.1:8000/docs


You’ll see the interactive API documentation (Swagger UI).

🧪 Running an Example Case

Navigate to /docs

Use the /run_case/{case_id} endpoint

Or upload files using /analyze_files

Example preloaded cases are loaded automatically on startup.

🛑 Stopping the Server

Press:

CTRL + C


Deactivate virtual environment:

deactivate

🧠 Tech Stack

Python 3.10+

FastAPI

Uvicorn

Pydantic

OpenAI API

YAML-based policy modeling