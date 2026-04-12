import os
from dotenv import load_dotenv
from groq import Groq

def load_api_key():
    
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not set in environment")

    return Groq(api_key=groq_api_key)