# ai/agents/core.py
from os import getenv

# Load environment variables
provider_name=getenv("PROVIDER_NAME", "openai")
model_name=getenv("MODEL_NAME", "gpt-3.5-turbo")


