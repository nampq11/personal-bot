# HealthService Recomendation Chatbot

# How to run
## Prerequisite
 - poetry 1.8.3

## Setup
 - Download dataset
 - Create a new .env file based on `.env.example` add populate the variables there
 - Install dependencies with Poetry: `poetry install`
 - Start the jupyter lab: `poetry run jupyter lab`

## Start chatbot UI
 - Navigate to `ui` folder: `cd ui`
 - Run: `poetry run chainlit run chat_app.py`
 - Access the Chatbot UI at http://localhost:8000
