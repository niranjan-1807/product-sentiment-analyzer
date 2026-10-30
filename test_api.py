import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key or api_key == "your_groq_api_key_here":
    print("ERROR: Please set your actual GROQ_API_KEY in the .env file!")
    exit(1)

client = Groq(api_key=api_key)

print("Testing connection to Groq API...")
try:
    # Use openai/gpt-oss-120b or qwen/qwen3.8-27b on Groq
    model_to_test = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    response = client.chat.completions.create(
        model=model_to_test,
        messages=[
            {"role": "user", "content": "Say hello in 5 words."}
        ],
        max_tokens=20
    )
    print("API Connection Successful!")
    print(f"Model used: {model_to_test}")
    print("Response:", response.choices[0].message.content)
except Exception as e:
    print("API Connection Failed:", str(e))
