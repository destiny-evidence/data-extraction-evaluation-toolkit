# Created by Lennart at 28/10/2026
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("AZURE_API_KEY"),
    base_url=os.environ.get("AZURE_API_BASE"),
)

response = client.responses.create(
    model="gpt-5.6-luna",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
)

print(response.output_text)

