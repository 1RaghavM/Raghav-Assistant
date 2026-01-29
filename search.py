import os
from dotenv import load_dotenv

load_dotenv()
import ollama


response = ollama.web_search("What is Ollama?")
print(response)

